using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using AutoMapper;
using bet_fred.Data;
using bet_fred.Models;
using bet_fred.DTOs;

namespace bet_fred.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class CustomersController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly IMapper _mapper;
        private readonly ILogger<CustomersController> _logger;

        public CustomersController(ApplicationDbContext context, IMapper mapper, ILogger<CustomersController> logger)
        {
            _context = context;
            _mapper = mapper;
            _logger = logger;
        }

        [HttpGet]
        public async Task<ActionResult<IEnumerable<CustomerDto>>> GetCustomers()
        {
            try
            {
                var customers = await _context.Customers.ToListAsync();
                var customerDtos = _mapper.Map<IEnumerable<CustomerDto>>(customers);
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
                var customer = await _context.Customers.FindAsync(id);

                if (customer == null)
                    return NotFound();

                return Ok(_mapper.Map<CustomerDto>(customer));
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
                var customer = _mapper.Map<Customer>(createDto);
                
                _context.Customers.Add(customer);
                await _context.SaveChangesAsync();
                
                var dto = _mapper.Map<CustomerDto>(customer);
                return CreatedAtAction(nameof(GetCustomer), new { id = customer.Id }, dto);
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
                var existing = await _context.Customers.FindAsync(id);
                if (existing == null)
                    return NotFound();

                _mapper.Map(updateDto, existing);
                
                await _context.SaveChangesAsync();

                return Ok(_mapper.Map<CustomerDto>(existing));
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
                var customer = await _context.Customers.FindAsync(id);
                
                if (customer == null)
                    return NotFound();

                _context.Customers.Remove(customer);
                await _context.SaveChangesAsync();

                return Ok("Customer deleted successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting customer {Id}", id);
                return StatusCode(500, "Error deleting customer");
            }
        }
    }
}
