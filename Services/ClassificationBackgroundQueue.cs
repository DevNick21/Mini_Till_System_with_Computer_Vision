using System.Threading.Channels;
using bet_fred.Data;
using Microsoft.EntityFrameworkCore;

namespace bet_fred.Services
{
    public interface IClassificationBackgroundQueue
    {
        void Enqueue(int betId, byte[] imageData);
    }

    internal record ClassificationWorkItem(int BetId, byte[] ImageData);

    public class ClassificationBackgroundQueue : IClassificationBackgroundQueue
    {
        private readonly Channel<ClassificationWorkItem> _channel;
        public ClassificationBackgroundQueue()
        {
            _channel = Channel.CreateUnbounded<ClassificationWorkItem>(new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = false
            });
        }

        public void Enqueue(int betId, byte[] imageData)
        {
            _channel.Writer.TryWrite(new ClassificationWorkItem(betId, imageData));
        }

        internal IAsyncEnumerable<ClassificationWorkItem> ReadAllAsync(CancellationToken token) => _channel.Reader.ReadAllAsync(token);
    }

    public class ClassificationProcessingService : BackgroundService
    {
        private readonly ILogger<ClassificationProcessingService> _logger;
        private readonly IServiceScopeFactory _scopeFactory;
        private readonly ClassificationBackgroundQueue _queue;
        private readonly IClassificationUpdateNotifier _notifier;

        public ClassificationProcessingService(ILogger<ClassificationProcessingService> logger,
                               IServiceScopeFactory scopeFactory,
                               IClassificationBackgroundQueue queue,
                               IClassificationUpdateNotifier notifier)
        {
            _logger = logger;
            _scopeFactory = scopeFactory;
            _queue = (ClassificationBackgroundQueue)queue;
            _notifier = notifier;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("ClassificationProcessingService started");
            await foreach (var work in _queue.ReadAllAsync(stoppingToken))
            {
                try
                {
                    using var scope = _scopeFactory.CreateScope();
                    var dataService = scope.ServiceProvider.GetRequiredService<IDataService>();
                    var classifier = scope.ServiceProvider.GetRequiredService<IClassificationService>();
                    var thresholdEvaluator = scope.ServiceProvider.GetRequiredService<IThresholdEvaluator>();
                    var db = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();

                    var bet = await db.BetRecords.FirstOrDefaultAsync(b => b.Id == work.BetId, stoppingToken);
                    if (bet == null)
                    {
                        _logger.LogWarning("Bet {BetId} missing during classification", work.BetId);
                        continue;
                    }
                    if (!string.IsNullOrEmpty(bet.WriterClassification))
                    {
                        _logger.LogDebug("Bet {BetId} already classified", work.BetId);
                        continue;
                    }

                    if (!await classifier.IsServiceHealthyAsync())
                    {
                        _logger.LogWarning("Classifier unhealthy; requeue bet {BetId}", work.BetId);
                        _queue.Enqueue(work.BetId, work.ImageData);
                        await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
                        continue;
                    }

                    var (writerId, confidence) = await classifier.ClassifyHandwritingAsync(work.ImageData, work.BetId);
                    if (writerId != null && confidence.HasValue)
                    {
                        await dataService.UpdateBetClassificationAsync(work.BetId, writerId, confidence.Value);
                        await _notifier.NotifyAsync(new ClassificationUpdate(work.BetId, writerId, confidence.Value));
                        var updated = await db.BetRecords.Include(b => b.Customer).FirstOrDefaultAsync(b => b.Id == work.BetId, stoppingToken);
                        if (updated != null && updated.CustomerId.HasValue)
                        {
                            await thresholdEvaluator.ProcessBetForThresholdsAsync(updated);
                        }
                        _logger.LogInformation("Background classification complete for bet {BetId}", work.BetId);
                    }
                    else
                    {
                        _logger.LogWarning("No classification result for bet {BetId}", work.BetId);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing classification work item");
                }
            }
        }
    }
}
