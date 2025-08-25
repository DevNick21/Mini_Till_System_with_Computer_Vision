using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using AutoMapper;
using bet_fred.Data;
using bet_fred.Services;
using bet_fred.Models;
using bet_fred.DTOs;

namespace bet_fred.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class BetController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly ClassificationService _classificationService;
        private readonly ThresholdEvaluator _thresholdEvaluator;
        private readonly IMapper _mapper;
        private readonly ILogger<BetController> _logger;
        private readonly IDataService _dataService;

        public BetController(
            ApplicationDbContext context,
            ClassificationService classificationService,
            ThresholdEvaluator thresholdEvaluator,
            IMapper mapper,
            ILogger<BetController> logger,
            IDataService dataService)
        {
            _context = context;
            _classificationService = classificationService;

            _thresholdEvaluator = thresholdEvaluator;
            _mapper = mapper;
            _logger = logger;
            _dataService = dataService;
        }

        [HttpGet]
        public async Task<ActionResult<IEnumerable<BetRecordDto>>> GetBets()
        {
            try
            {
                var bets = await _context.BetRecords
                    .Include(b => b.Customer)
                    .OrderByDescending(b => b.PlacedAt)
                    .ToListAsync();

                return Ok(_mapper.Map<IEnumerable<BetRecordDto>>(bets));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting bet records");
                return StatusCode(500, "Error retrieving bet records");
            }
        }

        [HttpGet("recent")]
        public async Task<ActionResult<IEnumerable<BetRecordDto>>> GetRecentBets()
        {
            try
            {
                var recentBets = await _context.BetRecords
                    .Include(b => b.Customer)
                    .OrderByDescending(b => b.PlacedAt)
                    .Take(20)
                    .ToListAsync();

                return Ok(_mapper.Map<IEnumerable<BetRecordDto>>(recentBets));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting recent bets");
                return StatusCode(500, "Error retrieving recent bets");
            }
        }

        [HttpGet("{id:int}")]
        public async Task<ActionResult<BetRecordDto>> GetBet(int id)
        {
            try
            {
                var bet = await _dataService.GetBetByIdAsync(id);

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
                if (string.IsNullOrWhiteSpace(createDto.ImageDataBase64))
                    return BadRequest("Slip image is required");

                byte[] imageBytes;
                try
                {
                    imageBytes = Convert.FromBase64String(createDto.ImageDataBase64);
                }
                catch
                {
                    return BadRequest("Invalid base64 slip image");
                }

                var betRecord = new BetRecord
                {
                    Amount = createDto.Amount,
                    CustomerId = createDto.CustomerId,
                    PlacedAt = DateTime.UtcNow,
                    ImageData = imageBytes
                };

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
                using var memoryStream = new MemoryStream();
                await file.CopyToAsync(memoryStream);
                var imageData = memoryStream.ToArray();

                decimal finalAmount = amount ?? 0;

                // Rely on provided amount or default to 0

                // Create a new bet record
                var newBet = new BetRecord
                {
                    Amount = finalAmount, // Amount from form
                    PlacedAt = DateTime.UtcNow,
                    CustomerId = customerId
                };

                var createdBet = await _dataService.CreateBetRecordAsync(newBet);

                // Persist the slip image to the bet record
                var stored = await _dataService.UploadSlipAsync(createdBet.Id, imageData);
                if (!stored)
                {
                    _logger.LogWarning("Failed to store slip image for bet {BetId}", createdBet.Id);
                }

                // Synchronous classification for MVP
                if (await _classificationService.IsServiceHealthyAsync())
                {
                    var (writerId, confidence) = await _classificationService.ClassifyHandwritingAsync(imageData, createdBet.Id);
                    if (writerId > 0 && confidence > 0.0)
                    {
                        await _dataService.UpdateBetClassificationAsync(createdBet.Id, writerId.ToString(), confidence);
                        
                        // Check if this writer is already linked to a customer
                        var existingCustomerBet = await _context.BetRecords
                            .Where(b => b.WriterClassification == writerId.ToString() && b.CustomerId != null)
                            .FirstOrDefaultAsync();
                        
                        if (existingCustomerBet != null)
                        {
                            // Link this bet to the same customer (without triggering threshold eval)
                            createdBet.CustomerId = existingCustomerBet.CustomerId;
                            await _context.SaveChangesAsync();
                        }
                    }
                }

                // Always evaluate thresholds for bets with writer classification
                var finalBet = await _dataService.GetBetByIdAsync(createdBet.Id);
                if (finalBet != null && !string.IsNullOrWhiteSpace(finalBet.WriterClassification))
                {
                    await _thresholdEvaluator.ProcessBetForThresholdsAsync(finalBet);
                }

                return Ok(MapToBetRecordDto(finalBet ?? createdBet));
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
                var existingBet = await _dataService.GetBetByIdAsync(id);

                if (existingBet == null)
                    return NotFound();

                // Update only provided properties
                if (updateDto.Amount.HasValue)
                    existingBet.Amount = updateDto.Amount.Value;

                // BetType removed

                // Outcome removed

                if (updateDto.CustomerId.HasValue)
                    existingBet.CustomerId = updateDto.CustomerId;

                var updatedBet = await _dataService.UpdateBetAsync(existingBet);

                // Handle classification updates separately
                if (updateDto.WriterClassification != null && updateDto.ClassificationConfidence.HasValue)
                {
                    await _dataService.UpdateBetClassificationAsync(id, updateDto.WriterClassification, updateDto.ClassificationConfidence.Value);
                    updatedBet = await _dataService.GetBetByIdAsync(id); // Re-fetch to get updated values
                }

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

                if (!await _dataService.UploadSlipAsync(id, imageData))
                    return BadRequest("Failed to upload slip - bet may not exist or already has image");

                // Image uploaded; no automatic amount extraction

                // Check if CV service is healthy before classification
                if (await _classificationService.IsServiceHealthyAsync())
                {
                    // Trigger CV service classification
                    var (writerId, confidence) = await _classificationService.ClassifyHandwritingAsync(imageData, id);

                    if (writerId > 0 && confidence > 0.0)
                    {
                        // Update bet record with classification results
                        await _dataService.UpdateBetClassificationAsync(id, writerId.ToString(), confidence);

                        // Check if this writer is already linked to a customer
                        var existingCustomerBet = await _context.BetRecords
                            .Where(b => b.WriterClassification == writerId.ToString() && b.CustomerId != null)
                            .FirstOrDefaultAsync();
                        
                        if (existingCustomerBet != null)
                        {
                            // Link this bet to the same customer (direct update to avoid recursion)
                            var betToUpdate = await _context.BetRecords.FindAsync(id);
                            if (betToUpdate != null)
                            {
                                betToUpdate.CustomerId = existingCustomerBet.CustomerId;
                                await _context.SaveChangesAsync();
                            }
                        }

                        // Get the updated bet record for threshold evaluation
                        var bet = await _dataService.GetBetByIdAsync(id);
                        if (bet != null && !string.IsNullOrWhiteSpace(bet.WriterClassification))
                        {
                            // Evaluate thresholds after classification
                            await _thresholdEvaluator.ProcessBetForThresholdsAsync(bet);
                        }
                        
                        if (bet != null)
                        {
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
                var currentBet = await _dataService.GetBetByIdAsync(id);
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
                WriterClassification = betRecord.WriterClassification,
                ClassificationConfidence = betRecord.ClassificationConfidence,
                CustomerId = betRecord.CustomerId,
                CustomerName = betRecord.Customer?.Name
            };
        }

        [HttpGet("{id:int}/status")]
        public async Task<ActionResult<object>> GetBetStatus(int id)
        {
            try
            {
                var bet = await _dataService.GetBetByIdAsync(id);
                if (bet == null) return NotFound();
                return Ok(new { bet.Id, bet.WriterClassification, bet.ClassificationConfidence });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting bet status {Id}", id);
                return StatusCode(500, "Error retrieving status");
            }
        }

        // DEV endpoint 'clear' removed to reduce surface area
    }
}
