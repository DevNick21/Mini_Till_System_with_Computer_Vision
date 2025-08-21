using Microsoft.AspNetCore.Mvc;
using bet_fred.Services;
using bet_fred.Models;

namespace bet_fred.Controllers
{
    [ApiController]
    [Route("api/ocr-suggestions")]
    public class OcrSuggestionsController : ControllerBase
    {
        private readonly IDataService _dataService;
        private readonly ILogger<OcrSuggestionsController> _logger;

        public OcrSuggestionsController(IDataService dataService, ILogger<OcrSuggestionsController> logger)
        {
            _dataService = dataService;
            _logger = logger;
        }

        public class CreateOcrSuggestionDto
        {
            public int? BetRecordId { get; set; }
            public string? FileName { get; set; }
            public long? FileSize { get; set; }
            public string? FileHash { get; set; }
            public decimal? Stake { get; set; }
            public string? Currency { get; set; }
            public string? Method { get; set; }
        }

        [HttpPost]
        public async Task<ActionResult<OcrSuggestion>> Create([FromBody] CreateOcrSuggestionDto dto)
        {
            try
            {
                var suggestion = new OcrSuggestion
                {
                    BetRecordId = dto.BetRecordId,
                    FileName = dto.FileName,
                    FileSize = dto.FileSize,
                    FileHash = dto.FileHash,
                    Stake = dto.Stake,
                    Currency = dto.Currency,
                    Method = dto.Method
                };

                var created = await _dataService.CreateOcrSuggestionAsync(suggestion);
                return CreatedAtAction(nameof(GetById), new { id = created.Id }, created);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating OCR suggestion");
                return StatusCode(500, "Error creating OCR suggestion");
            }
        }

        [HttpGet("{id:int}")]
        public async Task<ActionResult<OcrSuggestion>> GetById(int id)
        {
            var s = await _dataService.GetOcrSuggestionByIdAsync(id);
            if (s == null) return NotFound();
            return Ok(s);
        }

        [HttpPost("{id:int}/accept")]
        public async Task<ActionResult> Accept(int id)
        {
            try
            {
                var ok = await _dataService.MarkOcrSuggestionAcceptedAsync(id);
                if (!ok) return NotFound();
                return Ok(new { status = "accepted" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error marking OCR suggestion accepted {Id}", id);
                return StatusCode(500, "Error accepting suggestion");
            }
        }
    }
}
