using Microsoft.AspNetCore.Mvc;
using bet_fred.Services;

namespace bet_fred.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class SystemController : ControllerBase
    {
        private readonly IDataService _dataService;
        private readonly ILogger<SystemController> _logger;

        public SystemController(IDataService dataService, ILogger<SystemController> logger)
        {
            _dataService = dataService;
            _logger = logger;
        }

        [HttpPost("create-demo-data")]
        public async Task<ActionResult<string>> CreateDemoData()
        {
            try
            {
                var result = await _dataService.CreateDemoDataAsync();
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating demo data");
                return StatusCode(500, "An error occurred while creating demo data");
            }
        }

        [HttpPost("reset-database")]
        public async Task<ActionResult<string>> ResetDatabase()
        {
            try
            {
                var result = await _dataService.ResetDatabaseAsync();
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error resetting database");
                return StatusCode(500, "An error occurred while resetting database");
            }
        }

        [HttpPost("create-default-rules")]
        public async Task<ActionResult<string>> CreateDefaultRules()
        {
            try
            {
                var result = await _dataService.CreateDefaultRulesAsync();
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating default rules");
                return StatusCode(500, "An error occurred while creating default rules");
            }
        }

        [HttpGet("dashboard-stats")]
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
                return StatusCode(500, "An error occurred while retrieving dashboard stats");
            }
        }

        [HttpGet("alerts")]
        public async Task<ActionResult> GetAlerts()
        {
            try
            {
                var alerts = await _dataService.GetAlertsAsync();
                return Ok(alerts);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting alerts");
                return StatusCode(500, "An error occurred while retrieving alerts");
            }
        }
    }
}
