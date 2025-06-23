namespace bet_fred.Models
{
    /// <summary>
    /// Payload for PATCH /clusters/{id}/tag
    /// </summary>
    public record TagClusterRequest(int CustomerId);
}
