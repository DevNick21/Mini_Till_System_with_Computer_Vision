using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using bet_fred.Data;
using bet_fred.Models;

namespace bet_fred.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ThresholdsController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<ThresholdsController> _logger;

        public ThresholdsController(ApplicationDbContext context, ILogger<ThresholdsController> logger)
        {
            _context = context;
            _logger = logger;
        }

        public class CreateThresholdDto
        {
            public string Name { get; set; } = string.Empty;
            public string? Description { get; set; }
            public decimal Value { get; set; }
            public int TimeWindowMinutes { get; set; } = 1440;
            public bool IsActive { get; set; } = true;
        }

        public class UpdateThresholdDto
        {
            public string? Name { get; set; }
            public string? Description { get; set; }
            public decimal? Value { get; set; }
            public int? TimeWindowMinutes { get; set; }
            public bool? IsActive { get; set; }
        }

        [HttpGet]
        public async Task<ActionResult<IEnumerable<ThresholdRule>>> List()
        {
            var rules = await _context.ThresholdRules
                .OrderByDescending(r => r.IsActive)
                .ThenBy(r => r.Name)
                .ToListAsync();
            return Ok(rules);
        }

        [HttpGet("{id:int}")]
        public async Task<ActionResult<ThresholdRule>> Get(int id)
        {
            var rule = await _context.ThresholdRules.FindAsync(id);
            if (rule == null) return NotFound();
            return Ok(rule);
        }

        [HttpPost]
        public async Task<ActionResult<ThresholdRule>> Create([FromBody] CreateThresholdDto dto)
        {
            if (string.IsNullOrWhiteSpace(dto.Name))
                return BadRequest("Name is required");
            if (dto.Value < 0)
                return BadRequest("Value must be non-negative");
            if (dto.TimeWindowMinutes <= 0)
                return BadRequest("TimeWindowMinutes must be > 0");

            var rule = new ThresholdRule
            {
                Name = dto.Name.Trim(),
                Description = dto.Description?.Trim() ?? string.Empty,
                Value = dto.Value,
                TimeWindowMinutes = dto.TimeWindowMinutes,
                IsActive = dto.IsActive,
                CreatedAt = DateTime.UtcNow
            };

            _context.ThresholdRules.Add(rule);
            await _context.SaveChangesAsync();
            return CreatedAtAction(nameof(Get), new { id = rule.Id }, rule);
        }

        [HttpPut("{id:int}")]
        public async Task<ActionResult<ThresholdRule>> Update(int id, [FromBody] UpdateThresholdDto dto)
        {
            var rule = await _context.ThresholdRules.FindAsync(id);
            if (rule == null) return NotFound();

            if (dto.Name != null)
            {
                if (string.IsNullOrWhiteSpace(dto.Name))
                    return BadRequest("Name cannot be empty");
                rule.Name = dto.Name.Trim();
            }
            if (dto.Description != null)
                rule.Description = dto.Description.Trim();
            if (dto.Value.HasValue)
            {
                if (dto.Value.Value < 0)
                    return BadRequest("Value must be non-negative");
                rule.Value = dto.Value.Value;
            }
            if (dto.TimeWindowMinutes.HasValue)
            {
                if (dto.TimeWindowMinutes.Value <= 0)
                    return BadRequest("TimeWindowMinutes must be > 0");
                rule.TimeWindowMinutes = dto.TimeWindowMinutes.Value;
            }
            if (dto.IsActive.HasValue)
                rule.IsActive = dto.IsActive.Value;

            await _context.SaveChangesAsync();
            return Ok(rule);
        }

        [HttpPatch("{id:int}/toggle")]
        public async Task<ActionResult<ThresholdRule>> Toggle(int id)
        {
            var rule = await _context.ThresholdRules.FindAsync(id);
            if (rule == null) return NotFound();
            rule.IsActive = !rule.IsActive;
            await _context.SaveChangesAsync();
            return Ok(rule);
        }
    }
}
