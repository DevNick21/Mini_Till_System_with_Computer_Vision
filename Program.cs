using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.IdentityModel.Tokens;
// using OpenTelemetry.Resources;
// using OpenTelemetry.Trace;
// using OpenTelemetry.Metrics;
// using OpenTelemetry.Instrumentation.EntityFrameworkCore;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using Prometheus;
using System.Text;
using Serilog;
using bet_fred.Data;
using bet_fred.Models;
using bet_fred.Services;

var builder = WebApplication.CreateBuilder(args);

// EF-Core on SQLite
builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

// Configure maximum request body size (e.g., for image uploads)
builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 10 * 1024 * 1024; // 10 MB
});

// builder.Services.AddSingleton<IAntivirusScanner, ClamAVScanner>();

// Serilog for logging
Log.Logger = new LoggerConfiguration()
    .ReadFrom.Configuration(builder.Configuration)
    .Enrich.FromLogContext()
    .CreateLogger();

builder.Host.UseSerilog();

// // OpenTelemetry for tracing and metrics
// builder.Services.AddOpenTelemetry()
//     .WithTracing(tp =>
//     {
//         tp.AddAspNetCoreInstrumentation()
//           .AddHttpClientInstrumentation()
//           .AddEntityFrameworkCoreInstrumentation();
//     })
//     .WithMetrics(mp =>
//     {
//         mp
//          .AddAspNetCoreInstrumentation()
//          .AddHttpClientInstrumentation();
//     });

// Authentication
var jwtKey    = builder.Configuration.GetValue<string>("Jwt:Key")
                ?? throw new InvalidOperationException("Missing configuration: Jwt:Key");

var jwtIssuer = builder.Configuration["Jwt:Issuer"];
builder.Services
    .AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters {
            ValidateIssuer           = true,
            ValidateAudience         = true,
            ValidateLifetime         = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer              = jwtIssuer,
            ValidAudience            = jwtIssuer,
            IssuerSigningKey         = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey))
        };
    });
builder.Services.AddHttpClient("classification", client =>
{
    client.BaseAddress = new Uri("http://localhost:8001/");
    client.Timeout = TimeSpan.FromSeconds(60); // Classification might take longer
});
// API and Swagger
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddHostedService<ThresholdHostedService>();
builder.Services.AddScoped<ThresholdEvaluator>();

builder.Services.AddAuthorization();


var app = builder.Build();


app.UseSerilogRequestLogging();
app.UseAuthentication();
app.UseAuthorization();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseDefaultFiles();
app.UseStaticFiles();

// Map controllers and minimal endpoints
// Expose Prometheus metrics at /metrics
// app.UseMetricServer("/metrics");
app.MapControllers();
app.MapGet("/customers", async (ApplicationDbContext db) =>
    await db.Customers.ToListAsync());

app.MapFallbackToFile("index.html");


var users = new[] { new { Username = "admin", Password = "password" } };

app.MapPost("/login", (LoginRequest creds) =>
{
    var user = users.SingleOrDefault(u =>
        u.Username == creds.Username && u.Password == creds.Password
    );
    if (user is null)
        return Results.Unauthorized();

    // Create JWT
    var claims = new[]
    {
        new Claim(JwtRegisteredClaimNames.Sub, user.Username),
        new Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString())
    };
    var key    = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey));
    var credsT = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);
    var token  = new JwtSecurityToken(
        issuer: jwtIssuer,
        audience: jwtIssuer,
        claims: claims,
        expires: DateTime.UtcNow.AddHours(2),
        signingCredentials: credsT
    );
    var jwt = new JwtSecurityTokenHandler().WriteToken(token);
    return Results.Ok(new { token = jwt });
})
.Accepts<LoginRequest>("application/json")
.Produces(StatusCodes.Status200OK)
.Produces(StatusCodes.Status401Unauthorized);



// Creates a new BetRecord (metadata only)
app.MapPost("/bets", async (BetRecord record, ApplicationDbContext db) =>
{
    // record.Outcome will be Unknown, record.ImageData is empty
    db.BetRecords.Add(record);
    await db.SaveChangesAsync();
    return Results.Created($"/bets/{record.Id}", record);
})
.WithName("CreateBet")
.Produces<BetRecord>(StatusCodes.Status201Created);

// Uploads a slip for an existing Bet
app.MapPost("/bets/{id}/slip", async (int id, IFormFile file, ApplicationDbContext db) =>
{
    var bet = await db.BetRecords.FindAsync(id);
    if (bet is null)
        return Results.NotFound($"Bet {id} not found.");

    if (file == null)
        return Results.BadRequest("No file uploaded");

    // Validate type
    var allowed = new[] { "image/jpeg", "image/png" };
    if (!allowed.Contains(file.ContentType))
        return Results.BadRequest("Only JPEG/PNG");

    // Prevent double upload
    if (bet.ImageData?.Length > 0)
        return Results.Conflict("Slip already uploaded");

    using var ms = new MemoryStream();
    await file.CopyToAsync(ms);
    bet.ImageData = ms.ToArray();

    await db.SaveChangesAsync();
    return Results.Ok(bet);
})
.WithName("UploadSlip")
.AllowAnonymous()                // if you still want anonymous access
.DisableAntiforgery()            // ← disable the new built-in XSRF check
.Accepts<IFormFile>("multipart/form-data")
.Produces<BetRecord>(StatusCodes.Status200OK)
.Produces<string>(StatusCodes.Status400BadRequest)
.Produces<string>(StatusCodes.Status404NotFound)
.Produces<string>(StatusCodes.Status409Conflict);
// .RequireAuthorization();


// deletes a BetRecord by Id
app.MapDelete("/bets/{id}", async (int id, ApplicationDbContext db) =>
{
    var bet = await db.BetRecords.FindAsync(id);
    if (bet is null)
        return Results.NotFound($"No bet with Id {id}.");

    db.BetRecords.Remove(bet);
    await db.SaveChangesAsync();
    return Results.NoContent();
})
.WithName("DeleteBet")
.Produces(StatusCodes.Status204NoContent)
.Produces<string>(StatusCodes.Status404NotFound);


app.MapGet("/threshold-check", async (ThresholdEvaluator evaluator) =>
{
    var (pendingTags, alerts) = await evaluator.EvaluateAllThresholdsAsync();
    return Results.Ok(new {
        PendingTagsCreated = pendingTags,
        AlertsCreated = alerts
    });
})
.WithName("ThresholdCheck")
.Produces<List<PendingTag>>(StatusCodes.Status200OK);

// List all threshold rules
app.MapGet("/rules", async (ApplicationDbContext db) =>
    await db.ThresholdRules.ToListAsync())
   .WithName("GetRules")
   .Produces<List<ThresholdRule>>(StatusCodes.Status200OK);


app.MapGet("/pendingtags", async (ApplicationDbContext db) =>
    await db.PendingTags.ToListAsync());

// ADD - New endpoints for writer classification system
app.MapGet("/writer-classifications", async (ApplicationDbContext db) =>
    await db.WriterClassifications
        .Include(wc => wc.BetRecord)
        .Include(wc => wc.Customer)
        .Select(wc => new
        {
            wc.Id,
            wc.BetRecordId,
            wc.WriterId,
            wc.Confidence,
            wc.ConfidenceLevel,
            wc.CreatedAt,
            CustomerName = wc.Customer != null ? wc.Customer.TagName : null,
            BetAmount = wc.BetRecord.Amount
        })
        .ToListAsync())
.WithName("GetWriterClassifications");

// ADD - Tag a writer (create customer from pending tag)
app.MapPost("/pending-tags/{id}/complete", async (
    int id,
    CompletePendingTagRequest req,
    ApplicationDbContext db) =>
{
    var pendingTag = await db.PendingTags.FindAsync(id);
    if (pendingTag is null)
        return Results.NotFound($"Pending tag {id} not found.");

    if (pendingTag.IsCompleted)
        return Results.Conflict("Pending tag already completed.");

    // Create new Customer
    var customer = new Customer
    {
        TagName = req.CustomerName,
        FirstSeen = DateTime.UtcNow,
        IsTagged = true
    };
    db.Customers.Add(customer);
    await db.SaveChangesAsync(); // Get customer ID

    // Update all WriterClassifications for this writer
    var classifications = await db.WriterClassifications
        .Where(wc => wc.WriterId == pendingTag.WriterId)
        .ToListAsync();

    foreach (var classification in classifications)
    {
        classification.CustomerId = customer.Id;
    }

    // Mark pending tag as completed
    pendingTag.IsCompleted = true;
    pendingTag.CompletedAt = DateTime.UtcNow;

    await db.SaveChangesAsync();

    return Results.Ok(new
    {
        Customer = customer,
        UpdatedClassifications = classifications.Count,
        CompletedPendingTag = pendingTag
    });
})
.WithName("CompletePendingTag");

// Add this to your Program.cs after the existing endpoints

// Manual classification trigger (for testing)
app.MapPost("/classify-now", async (ApplicationDbContext db, IHttpClientFactory httpFactory) =>
{
    var unclassified = await db.BetRecords
        .Where(br => br.ImageData != null &&
                   !db.WriterClassifications.Any(wc => wc.BetRecordId == br.Id))
        .Take(10) // Limit for testing
        .ToListAsync();

    if (!unclassified.Any())
        return Results.Ok(new { message = "No unclassified records found" });

    // Call classification API
    using var client = httpFactory.CreateClient();
    using var content = new MultipartFormDataContent();

    foreach (var bet in unclassified)
    {
        var imageContent = new ByteArrayContent(bet.ImageData);
        imageContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");
        content.Add(imageContent, "files", $"{bet.Id}.jpg");
    }

    try
    {
        var response = await client.PostAsync("http://localhost:8001/classify-anonymous", content);
        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync();
            return Results.Problem($"Classification failed: {response.StatusCode} - {errorContent}");
        }

        var result = await response.Content.ReadFromJsonAsync<ClassificationApiResponse>();

        // Save results
        foreach (var classificationResult in result.Results)
        {
            var writerClassification = new WriterClassification
            {
                BetRecordId = classificationResult.SlipId,
                WriterId = classificationResult.WriterId,
                Confidence = classificationResult.Confidence,
                ConfidenceLevel = classificationResult.ConfidenceLevel,
                CreatedAt = DateTime.UtcNow
            };
            db.WriterClassifications.Add(writerClassification);
        }
        await db.SaveChangesAsync();

        return Results.Ok(new
        {
            ClassificationsCreated = result.Results.Count,
            Results = result.Results
        });
    }
    catch (HttpRequestException ex)
    {
        return Results.Problem($"Network error calling classification API: {ex.Message}");
    }
    catch (TaskCanceledException ex)
    {
        return Results.Problem($"Classification API timeout: {ex.Message}");
    }
    catch (Exception ex)
    {
        return Results.Problem($"Classification error: {ex.Message}");
    }
})
.WithName("ManualClassify");

// Health check for the classification API
app.MapGet("/health/classification", async (IHttpClientFactory httpFactory) =>
{
    try
    {
        using var client = httpFactory.CreateClient();
        var response = await client.GetAsync("http://localhost:8001/health");

        if (response.IsSuccessStatusCode)
        {
            var healthData = await response.Content.ReadFromJsonAsync<Dictionary<string, object>>();
            return Results.Ok(new {
                ClassificationAPI = "Healthy",
                Details = healthData
            });
        }

        return Results.Problem($"Classification API unhealthy: {response.StatusCode}");
    }
    catch (Exception ex)
    {
        return Results.Problem($"Cannot reach classification API: {ex.Message}");
    }
})
.WithName("ClassificationHealthCheck");

// Manually remove a pending‐tag
app.MapDelete("/pendingtags/{id}", async (int id, ApplicationDbContext db) =>
{
    var tag = await db.PendingTags.FindAsync(id);
    if (tag is null)
        return Results.NotFound($"No pending tag with Id {id}.");

    db.PendingTags.Remove(tag);
    await db.SaveChangesAsync();
    return Results.NoContent();
})
.WithName("DeletePendingTag")
.Produces(StatusCodes.Status204NoContent)
.Produces<string>(StatusCodes.Status404NotFound);

app.MapGet("/bets/{id}/slip", async (int id, ApplicationDbContext db) =>
{
    var bet = await db.BetRecords.FindAsync(id);
    if (bet is null || bet.ImageData == null || bet.ImageData.Length == 0)
        return Results.NotFound();

    // You could inspect the first few bytes to guess PNG vs JPEG,
    // but if you only allow those two types, JPEG is fine:
    return Results.File(bet.ImageData, "image/jpeg", $"{id}.jpg");
})
.WithName("GetSlipImage")
.Produces(StatusCodes.Status200OK)
.Produces(StatusCodes.Status404NotFound);

app.Run();

// Response models for the classification API
public class ClassificationApiResponse
{
    public List<ClassificationResult> Results { get; set; } = new();
    public Dictionary<string, object> Summary { get; set; } = new();
    public string Timestamp { get; set; } = string.Empty;
}

public class ClassificationResult
{
    public int SlipId { get; set; }
    public int WriterId { get; set; }
    public double Confidence { get; set; }
    public string ConfidenceLevel { get; set; } = string.Empty;
}

public record CompletePendingTagRequest(string CustomerName);
public record LoginRequest(string Username, string Password);

