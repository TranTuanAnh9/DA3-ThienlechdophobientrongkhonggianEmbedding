$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$sourcePath = 'D:\DA-3\pytest\slide-review-current\Bao Cao DoAn3 (1)-before-slide46.pptx'
$outputPath = 'D:\DA-3\Bao Cao DoAn3 (1)-updated-slide46.pptx'
$entryName = 'ppt/slides/slide46.xml'

Copy-Item -LiteralPath $sourcePath -Destination $outputPath -Force

$zip = [System.IO.Compression.ZipFile]::Open($outputPath, [System.IO.Compression.ZipArchiveMode]::Update)
try {
    $entry = $zip.GetEntry($entryName)
    $stream = $entry.Open()
    $reader = [System.IO.StreamReader]::new($stream)
    $xmlText = $reader.ReadToEnd()
    $reader.Close()
    $stream.Close()

    [xml]$xml = $xmlText
    $ns = [System.Xml.XmlNamespaceManager]::new($xml.NameTable)
    $ns.AddNamespace('p', 'http://schemas.openxmlformats.org/presentationml/2006/main')
    $ns.AddNamespace('a', 'http://schemas.openxmlformats.org/drawingml/2006/main')

    $shape = $xml.SelectSingleNode("//p:sp[p:nvSpPr/p:cNvPr[@name='TextBox 17']]", $ns)
    if (-not $shape) { throw 'Không tìm thấy TextBox 17 trên slide 46.' }

    $paragraphs = @($shape.SelectNodes('p:txBody/a:p', $ns))
    $keepIndices = @(1, 2, 3, 4, 5, 8, 9, 13, 14, 16, 17)
    $replacementText = @(
        'Kết luận phần liên hệ',
        '',
        'Có mối liên hệ',
        'Popular, medium và rare có geometry khác nhau; item gần và cùng hướng hơn với tâm category thường nhận nhiều exposure và có Hit@20 cao hơn.',
        '',
        'Geometry là yếu tố bổ sung',
        'Sau khi kiểm soát frequency, quan hệ vẫn tồn tại nhưng geometry chỉ tăng R² khoảng 0,7–1,2 điểm phần trăm; frequency vẫn là yếu tố mạnh hơn.',
        'Phạm vi kết luận',
        'Geometry gap không quyết định mô hình nào thiên lệch hơn; kết quả hiện tại chỉ cho thấy liên hệ thống kê, chưa chứng minh quan hệ nhân quả.',
        'KẾT LUẬN CHÍNH',
        'Embedding geometry có liên hệ với popularity bias, nhưng chỉ là yếu tố bổ sung bên cạnh train frequency và cơ chế dự đoán của từng mô hình.'
    )

    for ($i = 0; $i -lt $paragraphs.Count; $i++) {
        $oneBased = $i + 1
        if ($keepIndices -notcontains $oneBased) {
            [void]$paragraphs[$i].ParentNode.RemoveChild($paragraphs[$i])
        }
    }

    $remaining = @($shape.SelectNodes('p:txBody/a:p', $ns))
    if ($remaining.Count -ne $replacementText.Count) {
        throw "Số paragraph sau khi rút gọn không khớp: $($remaining.Count)."
    }

    for ($i = 0; $i -lt $remaining.Count; $i++) {
        $textNodes = @($remaining[$i].SelectNodes('.//a:t', $ns))
        if ($textNodes.Count -eq 0) {
            if ($replacementText[$i] -ne '') {
                throw "Paragraph $($i + 1) không có text node để thay thế."
            }
            continue
        }

        $textNodes[0].InnerText = $replacementText[$i]
        for ($j = 1; $j -lt $textNodes.Count; $j++) {
            $textNodes[$j].InnerText = ''
        }
    }

    $settings = [System.Xml.XmlWriterSettings]::new()
    $settings.Encoding = [System.Text.UTF8Encoding]::new($false)
    $settings.Indent = $false
    $settings.OmitXmlDeclaration = $false
    $builder = [System.Text.StringBuilder]::new()
    $writer = [System.Xml.XmlWriter]::Create($builder, $settings)
    $xml.Save($writer)
    $writer.Flush()
    $writer.Close()

    $entry.Delete()
    $newEntry = $zip.CreateEntry($entryName, [System.IO.Compression.CompressionLevel]::Optimal)
    $newStream = $newEntry.Open()
    $newWriter = [System.IO.StreamWriter]::new($newStream, [System.Text.UTF8Encoding]::new($false))
    $newWriter.Write($builder.ToString())
    $newWriter.Flush()
    $newWriter.Close()
    $newStream.Close()
}
finally {
    $zip.Dispose()
}

Write-Output $outputPath
