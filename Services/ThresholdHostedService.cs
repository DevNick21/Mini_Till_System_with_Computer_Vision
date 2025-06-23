using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;

namespace bet_fred.Services
{
    public class ThresholdHostedService : BackgroundService
    {
        private readonly IServiceProvider _sp;

        public ThresholdHostedService(IServiceProvider sp)
        {
            _sp = sp;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                using var scope = _sp.CreateScope();
                var evaluator = scope.ServiceProvider.GetRequiredService<ThresholdEvaluator>();
                await evaluator.EvaluateAsync();
                // wait before next check
                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken);
            }
        }
    }
}
