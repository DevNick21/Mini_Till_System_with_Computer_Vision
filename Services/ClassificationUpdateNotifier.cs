using System.Collections.Concurrent;
using System.Threading.Channels;
namespace bet_fred.Services
{
    public record ClassificationUpdate(int BetId, string WriterClassification, double Confidence);

    public interface IClassificationUpdateNotifier
    {
        (Guid Id, ChannelReader<ClassificationUpdate> Reader) Subscribe();
        void Unsubscribe(Guid id);
        Task NotifyAsync(ClassificationUpdate update);
    }

    public class ClassificationUpdateNotifier : IClassificationUpdateNotifier
    {
        private readonly ConcurrentDictionary<Guid, Channel<ClassificationUpdate>> _subscribers = new();
        private readonly ILogger<ClassificationUpdateNotifier> _logger;
        public ClassificationUpdateNotifier(ILogger<ClassificationUpdateNotifier> logger)
        {
            _logger = logger;
        }

        public (Guid Id, ChannelReader<ClassificationUpdate> Reader) Subscribe()
        {
            var id = Guid.NewGuid();
            var channel = Channel.CreateUnbounded<ClassificationUpdate>(new UnboundedChannelOptions { SingleWriter = false, SingleReader = false });
            _subscribers[id] = channel;
            _logger.LogInformation("SSE subscriber {SubscriberId} connected (total {Count})", id, _subscribers.Count);
            return (id, channel.Reader);
        }

        public void Unsubscribe(Guid id)
        {
            if (_subscribers.TryRemove(id, out var ch))
            {
                ch.Writer.TryComplete();
                _logger.LogInformation("SSE subscriber {SubscriberId} disconnected (total {Count})", id, _subscribers.Count);
            }
        }

        public async Task NotifyAsync(ClassificationUpdate update)
        {
            List<Guid> toRemove = new();
            foreach (var kvp in _subscribers)
            {
                if (!kvp.Value.Writer.TryWrite(update))
                {
                    toRemove.Add(kvp.Key);
                }
            }
            foreach (var id in toRemove)
            {
                Unsubscribe(id);
            }
            if (_subscribers.Count > 0)
            {
                _logger.LogInformation("Broadcast classification update for bet {BetId} to {Count} subscribers", update.BetId, _subscribers.Count);
            }
            await Task.CompletedTask;
        }
    }
}
