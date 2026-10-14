using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;

internal static class UninstallLauncher
{
    [STAThread]
    private static void Main()
    {
        string directory = AppDomain.CurrentDomain.BaseDirectory;
        string uninstaller = Directory.GetFiles(directory, "unins*.exe")
            .Where(path => Regex.IsMatch(Path.GetFileName(path), @"^unins[0-9]+\.exe$", RegexOptions.IgnoreCase))
            .OrderByDescending(path => path)
            .FirstOrDefault();

        if (uninstaller == null)
        {
            MessageBox.Show("Cannot find the installed uninstaller in this folder.",
                "Delta Exit Assistant", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        try
        {
            Process.Start(new ProcessStartInfo(uninstaller) { UseShellExecute = true });
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "Delta Exit Assistant",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}
