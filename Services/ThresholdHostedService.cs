using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using bet_fred.Services;

namespace bet_fred.Services
{
    public class ThresholdHostedService : BackgroundService
    {
        private readonly IServiceProvider _serviceProvider;
        private readonly ILogger<ThresholdHostedService> _logger;
        private readonly TimeSpan _interval = TimeSpan.FromMinutes(15); // Run every 15 minutes

        public ThresholdHostedService(
            IServiceProvider serviceProvider,
            ILogger<ThresholdHostedService> logger)
        {
            _serviceProvider = serviceProvider;
            _logger = logger;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("Threshold monitoring service is starting");

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    _logger.LogInformation("Running scheduled threshold evaluation");
                    await EvaluateThresholdsAsync();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error occurred during threshold evaluation");
                }

                _logger.LogInformation("Threshold evaluation complete, waiting {Interval} minutes until next run", _interval.TotalMinutes);
                await Task.Delay(_interval, stoppingToken);
            }

            _logger.LogInformation("Threshold monitoring service is stopping");
        }

        private async Task EvaluateThresholdsAsync()
        {
            // Create a scope to resolve scoped services
            using var scope = _serviceProvider.CreateScope();
            try
            {
                var evaluator = scope.ServiceProvider.GetRequiredService<IThresholdEvaluator>();
                var alerts = await evaluator.EvaluateThresholdsAsync();

                _logger.LogInformation("Generated {AlertCount} alerts during scheduled evaluation", alerts.Count());
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error resolving services for threshold evaluation");
            }
        }
    }
}
