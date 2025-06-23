using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.IdentityModel.Tokens;
// using OpenTelemetry.Resources;
// using OpenTelemetry.Trace;
// using OpenTelemetry.Metrics;
// using OpenTelemetry.Instrumentation.EntityFrameworkCore;
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

builder.Services.AddSingleton<IAntivirusScanner, ClamAVScanner>();

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

// Computer Vision API client
builder.Services.AddHttpClient("cv", client =>
{
    client.BaseAddress = new Uri("http://localhost:8000/");
    client.Timeout = TimeSpan.FromSeconds(30);
});

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

// API and Swagger
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddScoped<ThresholdEvaluator>();
builder.Services.AddHostedService<ThresholdHostedService>();
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
app.UseStaticFiles();

// Map controllers and minimal endpoints
// Expose Prometheus metrics at /metrics
// app.UseMetricServer("/metrics");
app.MapControllers();
app.MapGet("/customers", async (ApplicationDbContext db) =>
    await db.Customers.ToListAsync());

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
app.MapPost("/bets/{id}/slip", async (int id, IFormFile file, IAntivirusScanner av, ApplicationDbContext db) =>
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

    if (!await av.IsCleanAsync(bet.ImageData))
        return Results.BadRequest("Uploaded file failed virus scan.");

    await db.SaveChangesAsync();
    return Results.Ok(bet);
})
.WithName("UploadSlip")
.Accepts<IFormFile>("multipart/form-data")
.Produces<BetRecord>(StatusCodes.Status200OK)
.Produces<string>(StatusCodes.Status400BadRequest)
.Produces<string>(StatusCodes.Status404NotFound)
.Produces<string>(StatusCodes.Status409Conflict)
.RequireAuthorization();


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
    var created = await evaluator.EvaluateAsync();
    return Results.Ok(created);
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

// Cluster & record all slips
app.MapGet("/cluster-and-assign", async (ApplicationDbContext db, IHttpClientFactory httpFactory) =>
{
    // 1) Grab every BetRecord that’s got an image
    var batch = await db.BetRecords
        .Where(b => b.ImageData != null && b.ImageData.Length > 0)
        .Select(b => new { b.Id, ImageData = b.ImageData! })
        .ToListAsync();

    if (batch.Count == 0)
        return Results.BadRequest("No slips to cluster.");

    var already = new HashSet<int>(
        await db.HandwritingClusters
            .Select(h => h.BetRecordId)
            .ToListAsync()
   );
   var toCluster = batch.Where(b => !already.Contains(b.Id)).ToList();
   if (toCluster.Count == 0)
       return Results.Ok(new { message = "All slips already clustered" });

    // 2) Build a multipart/form-data request
    using var form = new MultipartFormDataContent();
    foreach (var b in batch)
    {
        var content = new ByteArrayContent(b.ImageData);
        form.Add(content, "files", $"{b.Id}.jpg");
    }

    // 3) Post to your Python CV service
    var client   = httpFactory.CreateClient("cv");
    var response = await client.PostAsync("/cluster", form);
    if (!response.IsSuccessStatusCode)
    {
        var error = await response.Content.ReadAsStringAsync();
        return Results.Problem(error, statusCode: (int)response.StatusCode);
    }

    var results = await response.Content.ReadFromJsonAsync<List<ClusterResult>>();
    if (results is null)
        return Results.Problem("Clustering service returned no data.");

    // 4) Insert a HandwritingCluster row per slip
    foreach (var r in results.Where(r => !already.Contains(r.Id)))
    {
        db.HandwritingClusters.Add(new HandwritingCluster {
            BetRecordId = r.Id,
            ClusterId   = r.Cluster,
            CreatedAt   = DateTime.UtcNow
            // CustomerId stays null until the cluster is tagged
        });
    }
    await db.SaveChangesAsync();

    // 5) Return the raw cluster assignments
    return Results.Ok(results);
})
.WithName("ClusterAndAssign")
.Produces<List<ClusterResult>>(StatusCodes.Status200OK);

app.MapGet("/clusters", async (ApplicationDbContext db) =>
    await db.HandwritingClusters
            .Select(h => new { h.Id, h.ClusterLabel, h.CreatedAt })
            .ToListAsync());

// Assign a real customer to a handwriting cluster, clear its pending tags, and fire alerts.
app.MapPatch("/clusters/{clusterId}/tag", async (
        int clusterId,
        TagClusterRequest req,
        ApplicationDbContext db) =>
{
    var cluster = await db.HandwritingClusters.FindAsync(clusterId);
    if (cluster is null)
        return Results.NotFound($"Cluster {clusterId} not found.");

    if (cluster.CustomerId != null)
        return Results.Conflict($"Cluster {clusterId} is already tagged to customer {cluster.CustomerId}.");

    // 1) Tie cluster to customer
    cluster.CustomerId = req.CustomerId;

    // 2) Find & remove all pending tags for this cluster
    var toRemove = await db.PendingTags
        .Where(t => t.Tag.EndsWith($"|{cluster.ClusterId}"))
        .ToListAsync();

    // 3) Create an Alert per removed tag
    var now = DateTime.UtcNow;
    var alerts = toRemove.Select(t =>
    {
        var ruleName = t.Tag.Split('|', 2)[0];
        return new Alert
        {
            CustomerId = req.CustomerId,
            Message = $"Customer {req.CustomerId} triggered rule '{ruleName}'.",
            TriggeredAt = now,
            RuleId = null  // or look up rule by name if you added RuleId to PendingTag
        };
    }).ToList();

    db.Alerts.AddRange(alerts);
    db.PendingTags.RemoveRange(toRemove);
    await db.SaveChangesAsync();

    return Results.Ok(new
    {
        Cluster = new { cluster.Id, cluster.ClusterId, cluster.CustomerId },
        alerts
    });
})
.WithName("TagCluster")
.Produces(StatusCodes.Status200OK)
.Produces<string>(StatusCodes.Status404NotFound)
.Produces<string>(StatusCodes.Status409Conflict);

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

app.Run();
