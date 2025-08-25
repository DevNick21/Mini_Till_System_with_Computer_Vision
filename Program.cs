using bet_fred.Data;
using bet_fred.Services;
using Microsoft.EntityFrameworkCore;
using Serilog;

namespace bet_fred
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add Serilog
            builder.Host.UseSerilog((context, config) =>
            {
                config.WriteTo.Console()
                      .WriteTo.File("Logs/log-.txt", rollingInterval: RollingInterval.Day);
            });

            // Add services to the container
            builder.Services.AddDbContext<ApplicationDbContext>(options =>
                options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection") ?? "Data Source=betfred.db"));

            // Add AutoMapper
            builder.Services.AddAutoMapper(typeof(Program));

            // Register simplified services
            builder.Services.AddScoped<IDataService, DataService>();
            builder.Services.AddScoped<ThresholdEvaluator>();
            builder.Services.AddHttpClient<ClassificationService>();
            // OCR service removed

            // Background service for threshold monitoring
            builder.Services.AddHostedService<ThresholdHostedService>();

            // Add API controllers with JSON configuration
            builder.Services.AddControllers()
                .AddJsonOptions(options =>
                {
                    // Prevent circular references in JSON serialization
                    options.JsonSerializerOptions.ReferenceHandler = System.Text.Json.Serialization.ReferenceHandler.IgnoreCycles;
                    options.JsonSerializerOptions.DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull;
                });
            
            // Minimal built-in health checks
            builder.Services.AddHealthChecks();

            // Add CORS to allow the React app to make API requests
            builder.Services.AddCors(options =>
            {
                options.AddPolicy("ReactAppPolicy", policy =>
                {
                    policy.SetIsOriginAllowed(origin =>
                    {
                        if (string.IsNullOrEmpty(origin)) return false;
                        var uri = new Uri(origin);
                        var isLocalhost = uri.Host == "localhost";
                        var validPorts = new[] { 3000, 5113 };
                        var validPort = validPorts.Contains(uri.Port);
                        return isLocalhost && validPort;
                    })
                    .AllowAnyMethod()
                    .AllowAnyHeader()
                    .AllowCredentials();
                });
            });

            // Add SPA static files
            builder.Services.AddSpaStaticFiles(configuration =>
            {
                configuration.RootPath = "frontend/build";
            });

            var app = builder.Build();

            // Configure the HTTP request pipeline
            if (!app.Environment.IsDevelopment())
            {
                app.UseHsts();
            }

            app.UseHttpsRedirection();
            app.UseStaticFiles();
            app.UseRouting();
            // Apply CORS after routing and before endpoints per ASP.NET Core guidance
            app.UseCors("ReactAppPolicy");

            // Apply pending migrations
            using (var scope = app.Services.CreateScope())
            {
                var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
                context.Database.Migrate();
            }

            // Map API controllers
            app.MapControllers();

            // Health endpoints (simple, no custom controller)
            app.MapHealthChecks("/health");
            // Back-compat for previous route used by HealthController
            app.MapHealthChecks("/api/health");

            // For development, just serve static files if they exist, otherwise return 404
            // The React dev server should run separately on port 3000
            if (!app.Environment.IsDevelopment())
            {
                app.UseSpaStaticFiles();
                app.MapFallbackToFile("index.html");
            }

            app.Run();
        }
    }
}
