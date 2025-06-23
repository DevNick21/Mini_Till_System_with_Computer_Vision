// bet_fred/Services/AntivirusScanner.cs
using System.IO;
using System.Threading.Tasks;
using nClam;  // dotnet add package nClam

public interface IAntivirusScanner
{
    Task<bool> IsCleanAsync(byte[] data);
}

public class ClamAVScanner : IAntivirusScanner
{
    private readonly ClamClient _clam;
    public ClamAVScanner(IConfiguration config)
    {
        var host = config["ClamAV:Host"] ?? "localhost";
        var port = int.Parse(config["ClamAV:Port"] ?? "3310");
        _clam = new ClamClient(host, port);
    }

    public async Task<bool> IsCleanAsync(byte[] data)
    {
        var result = await _clam.SendAndScanFileAsync(data);
        return result.Result == ClamScanResults.Clean;
    }
}
