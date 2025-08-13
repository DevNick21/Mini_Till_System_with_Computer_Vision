using bet_fred.Data;
using bet_fred.Services;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.SpaServices.ReactDevelopmentServer;
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

            // Register services (lean MVP)
            builder.Services.AddScoped<IDataService, DataService>();
            builder.Services.AddScoped<IThresholdEvaluator, ThresholdEvaluator>();

            // Register HttpClient for CV service
            builder.Services.AddHttpClient<IClassificationService, ClassificationService>();

            // ThresholdHostedService disabled for MVP

            // Add default configuration for Classification API if not present
            if (!builder.Configuration.GetSection("ClassificationApi").Exists())
            {
                builder.Configuration["ClassificationApi:BaseUrl"] = "http://localhost:8001";
            }

            // Add API controllers
            builder.Services.AddControllers();

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
            app.Use(async (context, next) =>
            {
                var origin = context.Request.Headers["Origin"].FirstOrDefault();
                if (!string.IsNullOrEmpty(origin))
                {
                    Console.WriteLine($"Incoming Origin: {origin}");
                }
                await next();
            });
            app.UseCors("ReactAppPolicy");

            // Apply pending migrations
            using (var scope = app.Services.CreateScope())
            {
                var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
                context.Database.Migrate();
            }

            // Map API controllers
            app.MapControllers();

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
