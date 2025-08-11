using Microsoft.AspNetCore.Mvc;
using bet_fred.Services;
using bet_fred.Models;
using bet_fred.DTOs;
using System.Linq;
using System.Text;

namespace bet_fred.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class BetController : ControllerBase
    {
        private readonly IDataService _dataService;
        private readonly IClassificationService _classificationService;
        private readonly IThresholdEvaluator _thresholdEvaluator;
        private readonly ILogger<BetController> _logger;

        public BetController(
            IDataService dataService,
            IClassificationService classificationService,
            IThresholdEvaluator thresholdEvaluator,
            ILogger<BetController> logger)
        {
            _dataService = dataService;
            _classificationService = classificationService;
            _thresholdEvaluator = thresholdEvaluator;
            _logger = logger;
        }

        [HttpGet]
        public async Task<ActionResult<IEnumerable<BetRecordDto>>> GetBets()
        {
            try
            {
                var bets = await _dataService.GetBetRecordsAsync();
                var betDtos = bets.Select(b => MapToBetRecordDto(b)).ToList();
                return Ok(betDtos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting bet records");
                return StatusCode(500, "Error retrieving bet records");
            }
        }

        [HttpGet("{id:int}")]
        public async Task<ActionResult<BetRecordDto>> GetBet(int id)
        {
            try
            {
                var bets = await _dataService.GetBetRecordsAsync();
                var bet = bets.FirstOrDefault(b => b.Id == id);

                if (bet == null)
                    return NotFound();

                return Ok(MapToBetRecordDto(bet));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting bet record {Id}", id);
                return StatusCode(500, "Error retrieving bet record");
            }
        }

        [HttpPost]
        public async Task<ActionResult<BetRecordDto>> CreateBet(CreateBetRecordDto createDto)
        {
            try
            {
                var betRecord = new BetRecord
                {
                    Amount = createDto.Amount,
                    CustomerId = createDto.CustomerId,
                    PlacedAt = DateTime.UtcNow
                };

                // Handle image data if provided
                if (!string.IsNullOrEmpty(createDto.ImageDataBase64))
                {
                    betRecord.ImageData = Convert.FromBase64String(createDto.ImageDataBase64);
                }

                var created = await _dataService.CreateBetRecordAsync(betRecord);
                return CreatedAtAction(nameof(GetBet), new { id = created.Id }, MapToBetRecordDto(created));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating bet record");
                return StatusCode(500, "Error creating bet record");
            }
        }

        [HttpPost("upload")]
        public async Task<ActionResult<BetRecordDto>> UploadBetImage(IFormFile file, [FromForm] int? customerId = null, [FromForm] decimal? amount = null)
        {
            if (file == null || file.Length == 0)
                return BadRequest("No file provided");

            try
            {
                // First create a new bet record
                var newBet = new BetRecord
                {
                    Amount = amount ?? 0, // Amount provided at upload time (stake)
                    PlacedAt = DateTime.UtcNow,
                    CustomerId = customerId
                };

                var createdBet = await _dataService.CreateBetRecordAsync(newBet);

                // Then upload the image for this bet
                using var memoryStream = new MemoryStream();
                await file.CopyToAsync(memoryStream);
                var imageData = memoryStream.ToArray();

                var success = await _dataService.UploadSlipAsync(createdBet.Id, imageData);

                if (!success)
                    return BadRequest("Failed to upload slip");

                // Process with CV service if available
                if (await _classificationService.IsServiceHealthyAsync())
                {
                    var (writerId, confidence) = await _classificationService.ClassifyHandwritingAsync(imageData, createdBet.Id);
                    if (writerId != null && confidence.HasValue)
                    {
                        await _dataService.UpdateBetClassificationAsync(createdBet.Id, writerId, confidence.Value);

                        // Get the updated bet for the response
                        var bets = await _dataService.GetBetRecordsAsync();
                        var updatedBet = bets.FirstOrDefault(b => b.Id == createdBet.Id);
                        if (updatedBet != null)
                        {
                            return Ok(MapToBetRecordDto(updatedBet));
                        }
                    }
                }

                // Return the bet
                return Ok(MapToBetRecordDto(createdBet));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error uploading bet image");
                return StatusCode(500, "Error uploading bet image");
            }
        }

        [HttpPut("{id}")]
        public async Task<ActionResult<BetRecordDto>> UpdateBet(int id, UpdateBetRecordDto updateDto)
        {
            try
            {
                // Get existing bet
                var bets = await _dataService.GetBetRecordsAsync();
                var existingBet = bets.FirstOrDefault(b => b.Id == id);

                if (existingBet == null)
                    return NotFound();

                // Update only provided properties
                if (updateDto.Amount.HasValue)
                    existingBet.Amount = updateDto.Amount.Value;

                // BetType removed

                if (updateDto.Outcome != null)
                {
                    if (Enum.TryParse<BetRecord.BetOutcome>(updateDto.Outcome, true, out var outcome))
                        existingBet.Outcome = outcome;
                }

                if (updateDto.CustomerId.HasValue)
                    existingBet.CustomerId = updateDto.CustomerId;

                if (updateDto.WriterClassification != null)
                    existingBet.WriterClassification = updateDto.WriterClassification;

                if (updateDto.ClassificationConfidence.HasValue)
                    existingBet.ClassificationConfidence = updateDto.ClassificationConfidence;

                var updatedBet = await _dataService.UpdateBetAsync(existingBet);

                if (updatedBet == null)
                    return NotFound();

                // Evaluate thresholds if applicable
                if (updatedBet.CustomerId.HasValue)
                {
                    await _thresholdEvaluator.ProcessBetForThresholdsAsync(updatedBet);
                }

                return Ok(MapToBetRecordDto(updatedBet));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating bet record {Id}", id);
                return StatusCode(500, "Error updating bet record");
            }
        }

        [HttpPost("{id}/upload-slip")]
        public async Task<ActionResult<BetRecordDto>> UploadSlip(int id, IFormFile file)
        {
            if (file == null || file.Length == 0)
                return BadRequest("No file provided");

            try
            {
                using var memoryStream = new MemoryStream();
                await file.CopyToAsync(memoryStream);
                var imageData = memoryStream.ToArray();

                var success = await _dataService.UploadSlipAsync(id, imageData);

                if (!success)
                    return BadRequest("Failed to upload slip - bet may not exist or already has image");

                // Check if CV service is healthy before classification
                if (await _classificationService.IsServiceHealthyAsync())
                {
                    // Trigger CV service classification
                    var (writerId, confidence) = await _classificationService.ClassifyHandwritingAsync(imageData, id);

                    if (writerId != null && confidence.HasValue)
                    {
                        // Update bet record with classification results
                        await _dataService.UpdateBetClassificationAsync(id, writerId, confidence.Value);

                        // Get the updated bet record for threshold evaluation
                        var bets = await _dataService.GetBetRecordsAsync();
                        var bet = bets.FirstOrDefault(b => b.Id == id);
                        if (bet != null && bet.CustomerId.HasValue)
                        {
                            // Evaluate thresholds after classification
                            await _thresholdEvaluator.ProcessBetForThresholdsAsync(bet);

                            return Ok(MapToBetRecordDto(bet));
                        }
                    }
                    else
                    {
                        _logger.LogWarning("Classification failed for bet {BetId}", id);
                    }
                }
                else
                {
                    _logger.LogWarning("CV service is unavailable, skipping classification for bet {BetId}", id);
                }

                // If we haven't returned yet, get the bet and return it
                var allBets = await _dataService.GetBetRecordsAsync();
                var currentBet = allBets.FirstOrDefault(b => b.Id == id);
                if (currentBet != null)
                {
                    return Ok(MapToBetRecordDto(currentBet));
                }

                return Ok("Slip uploaded successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error uploading slip for bet {Id}", id);
                return StatusCode(500, "Error uploading slip");
            }
        }

        [HttpGet("recent")]
        public async Task<ActionResult<IEnumerable<BetRecordDto>>> GetRecentBets()
        {
            try
            {
                var allBets = await _dataService.GetBetRecordsAsync();
                var recentBets = allBets.OrderByDescending(b => b.PlacedAt).Take(10).ToList();
                var recentDtos = recentBets.Select(b => MapToBetRecordDto(b)).ToList();
                return Ok(recentDtos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting recent bet records");
                return StatusCode(500, "Error retrieving recent bet records");
            }
        }

        [HttpGet("{id}/slip-image")]
        public async Task<ActionResult> GetSlipImage(int id)
        {
            try
            {
                var imageData = await _dataService.GetSlipImageAsync(id);

                if (imageData == null)
                    return NotFound();

                return File(imageData, "image/jpeg");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting slip image for bet {Id}", id);
                return StatusCode(500, "Error retrieving slip image");
            }
        }

        [HttpDelete("{id}")]
        public async Task<ActionResult> DeleteBet(int id)
        {
            try
            {
                var success = await _dataService.DeleteBetRecordAsync(id);

                if (!success)
                    return NotFound();

                return Ok("Bet record deleted successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting bet record {Id}", id);
                return StatusCode(500, "Error deleting bet record");
            }
        }

        /// <summary>
        /// Maps a BetRecord entity to a BetRecordDto
        /// </summary>
        private BetRecordDto MapToBetRecordDto(BetRecord betRecord)
        {
            return new BetRecordDto
            {
                Id = betRecord.Id,
                Amount = betRecord.Amount,
                PlacedAt = betRecord.PlacedAt,
                Outcome = betRecord.Outcome.ToString(),
                WriterClassification = betRecord.WriterClassification,
                ClassificationConfidence = betRecord.ClassificationConfidence,
                CustomerId = betRecord.CustomerId,
                CustomerName = betRecord.Customer?.Name
            };
        }
    }
}
