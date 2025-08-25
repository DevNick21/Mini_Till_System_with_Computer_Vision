using AutoMapper;
using bet_fred.Models;
using bet_fred.DTOs;

namespace bet_fred.Mappings
{
    public class MappingProfile : Profile
    {
        public MappingProfile()
        {
            // Customer mappings
            CreateMap<Customer, CustomerDto>();
            CreateMap<CreateCustomerDto, Customer>();
            CreateMap<UpdateCustomerDto, Customer>()
                .ForAllMembers(opt => opt.Condition((src, dest, srcMember) => srcMember != null));

            // BetRecord mappings
            CreateMap<BetRecord, BetRecordDto>()
                .ForMember(dest => dest.CustomerName, opt => opt.MapFrom(src => src.Customer != null ? src.Customer.Name : null));

        }
    }
}