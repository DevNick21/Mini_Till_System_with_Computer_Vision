using Microsoft.AspNetCore.Mvc;
using bet_fred.Services;
using bet_fred.Models;
using bet_fred.Data;

namespace bet_fred.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class SystemController : ControllerBase
    {
        private readonly IDataService _dataService;
        private readonly ThresholdEvaluator _thresholdEvaluator;
        private readonly ApplicationDbContext _context;
        private readonly ILogger<SystemController> _logger;

        public SystemController(IDataService dataService, ThresholdEvaluator thresholdEvaluator, ApplicationDbContext context, ILogger<SystemController> logger)
        {
            _dataService = dataService;
            _thresholdEvaluator = thresholdEvaluator;
            _context = context;
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

        [HttpPost("evaluate-thresholds")]
        public async Task<ActionResult<string>> EvaluateThresholds()
        {
            try
            {
                var alerts = await _thresholdEvaluator.EvaluateAllThresholdsAsync();
                return Ok($"Generated {alerts.Count} alerts");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error evaluating thresholds");
                return StatusCode(500, "An error occurred while evaluating thresholds");
            }
        }

        [HttpPost("create-test-alert")]
        public async Task<ActionResult<string>> CreateTestAlert()
        {
            try
            {
                var testAlert = new Alert
                {
                    AlertType = "TestAlert",
                    Message = "This is a test alert to verify the alert system is working",
                    CreatedAt = DateTime.UtcNow,
                    IsResolved = false
                };

                _context.Alerts.Add(testAlert);
                await _context.SaveChangesAsync();

                return Ok("Test alert created successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating test alert");
                return StatusCode(500, "An error occurred while creating test alert");
            }
        }
    }
}
