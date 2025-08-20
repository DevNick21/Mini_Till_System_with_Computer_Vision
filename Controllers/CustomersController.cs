using Microsoft.AspNetCore.Mvc;
using bet_fred.Services;
using bet_fred.Models;
using bet_fred.DTOs;
using System.Linq;

namespace bet_fred.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class CustomersController : ControllerBase
    {
        private readonly IDataService _dataService;
        private readonly ILogger<CustomersController> _logger;

        public CustomersController(IDataService dataService, ILogger<CustomersController> logger)
        {
            _dataService = dataService;
            _logger = logger;
        }

        [HttpGet]
        public async Task<ActionResult<IEnumerable<CustomerDto>>> GetCustomers()
        {
            try
            {
                var customers = await _dataService.GetCustomersAsync();
                var customerDtos = customers.Select(c => MapToCustomerDto(c)).ToList();
                return Ok(customerDtos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting customers");
                return StatusCode(500, "Error retrieving customers");
            }
        }

        [HttpGet("{id}")]
        public async Task<ActionResult<CustomerDto>> GetCustomer(int id)
        {
            try
            {
                var customer = await _dataService.GetCustomerByIdAsync(id);

                if (customer == null)
                    return NotFound();

                return Ok(MapToCustomerDto(customer));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting customer {Id}", id);
                return StatusCode(500, "Error retrieving customer");
            }
        }

        [HttpPost]
        public async Task<ActionResult<CustomerDto>> CreateCustomer(CreateCustomerDto createDto)
        {
            try
            {
                var customer = new Customer
                {
                    Name = createDto.Name
                };

                var created = await _dataService.CreateCustomerAsync(customer);
                var dto = MapToCustomerDto(created);
                return CreatedAtAction(nameof(GetCustomer), new { id = created.Id }, dto);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating customer");
                return StatusCode(500, "Error creating customer");
            }
        }

        [HttpPut("{id}")]
        public async Task<ActionResult<CustomerDto>> UpdateCustomer(int id, UpdateCustomerDto updateDto)
        {
            try
            {
                // First, get the existing customer
                var existing = await _dataService.GetCustomerByIdAsync(id);
                if (existing == null)
                    return NotFound();

                // Update only the properties that are provided
                if (updateDto.Name != null) existing.Name = updateDto.Name;

                var updated = await _dataService.UpdateCustomerAsync(id, existing);

                if (updated == null)
                    return NotFound();

                return Ok(MapToCustomerDto(updated));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating customer {Id}", id);
                return StatusCode(500, "Error updating customer");
            }
        }

        [HttpDelete("{id}")]
        public async Task<ActionResult> DeleteCustomer(int id)
        {
            try
            {
                var success = await _dataService.DeleteCustomerAsync(id);

                if (!success)
                    return NotFound();

                return Ok("Customer deleted successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting customer {Id}", id);
                return StatusCode(500, "Error deleting customer");
            }
        }

        /// <summary>
        /// Maps a Customer entity to a CustomerDto
        /// </summary>
        private CustomerDto MapToCustomerDto(Customer customer)
        {
            return new CustomerDto
            {
                Id = customer.Id,
                Name = customer.Name
            };
        }
    }
}
