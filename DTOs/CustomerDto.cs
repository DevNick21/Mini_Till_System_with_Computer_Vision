using System.ComponentModel.DataAnnotations;

namespace bet_fred.DTOs
{
    /// <summary>
    /// Data Transfer Object for Customer information
    /// </summary>
    public class CustomerDto
    {
        public int Id { get; set; }

        [Required]
        [StringLength(100)]
        public string Name { get; set; } = string.Empty;
    }

    /// <summary>
    /// DTO for creating a new customer
    /// </summary>
    public class CreateCustomerDto
    {
        [Required]
        [StringLength(100)]
        public string Name { get; set; } = string.Empty;
    }

    /// <summary>
    /// DTO for updating customer information
    /// </summary>
    public class UpdateCustomerDto
    {
        [StringLength(100)]
        public string? Name { get; set; }
    }
}