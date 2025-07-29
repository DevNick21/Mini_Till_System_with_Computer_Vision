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

        [EmailAddress]
        [StringLength(150)]
        public string Email { get; set; } = string.Empty;

        [Phone]
        [StringLength(20)]
        public string Phone { get; set; } = string.Empty;

        [StringLength(200)]
        public string Address { get; set; } = string.Empty;

        /// <summary>
        /// Maximum betting limit for fraud detection
        /// </summary>
        public decimal BetLimit { get; set; } = 1000;

        /// <summary>
        /// Risk assessment level (Low, Medium, High)
        /// </summary>
        [StringLength(20)]
        public string RiskLevel { get; set; } = "Low";

        // Summary statistics that don't expose the actual bet records
        public int TotalBets { get; set; }
        public decimal TotalStake { get; set; }
    }

    /// <summary>
    /// DTO for creating a new customer
    /// </summary>
    public class CreateCustomerDto
    {
        [Required]
        [StringLength(100)]
        public string Name { get; set; } = string.Empty;

        [EmailAddress]
        [StringLength(150)]
        public string Email { get; set; } = string.Empty;

        [Phone]
        [StringLength(20)]
        public string Phone { get; set; } = string.Empty;

        [StringLength(200)]
        public string Address { get; set; } = string.Empty;

        public decimal BetLimit { get; set; } = 1000;
    }

    /// <summary>
    /// DTO for updating customer information
    /// </summary>
    public class UpdateCustomerDto
    {
        [StringLength(100)]
        public string? Name { get; set; }

        [EmailAddress]
        [StringLength(150)]
        public string? Email { get; set; }

        [Phone]
        [StringLength(20)]
        public string? Phone { get; set; }

        [StringLength(200)]
        public string? Address { get; set; }

        public decimal? BetLimit { get; set; }

        [StringLength(20)]
        public string? RiskLevel { get; set; }
    }
}