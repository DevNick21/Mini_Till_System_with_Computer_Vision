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

            // Register services
            builder.Services.AddScoped<IDataService, DataService>();
            builder.Services.AddScoped<IThresholdEvaluator, ThresholdEvaluator>();

            // Register HttpClient for CV service
            builder.Services.AddHttpClient<IClassificationService, ClassificationService>();

            // Register background service for threshold monitoring
            builder.Services.AddHostedService<ThresholdHostedService>();

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
                options.AddPolicy("ReactAppPolicy", builder =>
                {
                    builder.WithOrigins("http://localhost:3000")
                           .AllowAnyMethod()
                           .AllowAnyHeader();
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
            app.UseCors("ReactAppPolicy");
            app.UseRouting();

            app.MapControllers();

            // Ensure database is created
            using (var scope = app.Services.CreateScope())
            {
                var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
                context.Database.EnsureCreated();
            }

            // Configure SPA serving
            app.UseSpa(spa =>
            {
                spa.Options.SourcePath = "frontend";

                if (app.Environment.IsDevelopment())
                {
                    spa.UseProxyToSpaDevelopmentServer("http://localhost:3000");
                }
            });

            app.Run();
        }
    }
}
