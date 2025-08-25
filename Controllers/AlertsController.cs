using Microsoft.AspNetCore.Mvc;
using bet_fred.Services;
using bet_fred.Models;

namespace bet_fred.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AlertsController : ControllerBase
    {
        private readonly IDataService _dataService;
        private readonly ILogger<AlertsController> _logger;

        public AlertsController(IDataService dataService, ILogger<AlertsController> logger)
        {
            _dataService = dataService;
            _logger = logger;
        }

        public class ResolveAlertRequest
        {
            public string? ResolvedBy { get; set; }
            public string? Notes { get; set; }
            public int? CustomerId { get; set; }
        }

        [HttpPost("{id:int}/resolve")]
        public async Task<ActionResult<Alert>> ResolveAlert(int id, [FromBody] ResolveAlertRequest body)
        {
            try
            {
                var alert = await _dataService.ResolveAlertAsync(id, body?.ResolvedBy, body?.Notes, body?.CustomerId);
                if (alert == null) return NotFound();
                return Ok(alert);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error resolving alert {Id}", id);
                return StatusCode(500, "Error resolving alert");
            }
        }

        [HttpGet]
        public async Task<ActionResult<IEnumerable<Alert>>> GetAlerts()
        {
            try
            {
                var alerts = await _dataService.GetAlertsAsync();
                return Ok(alerts);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting alerts");
                return StatusCode(500, "Error retrieving alerts");
            }
        }

        [HttpGet("dashboard")]
        public async Task<ActionResult<object>> GetDashboardStats()
        {
            try
            {
                var stats = await _dataService.GetDashboardStatsAsync();
                return Ok(stats);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting dashboard stats");
                return StatusCode(500, "Error retrieving dashboard statistics");
            }
        }
    }
}
