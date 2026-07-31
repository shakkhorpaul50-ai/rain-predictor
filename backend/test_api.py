import urllib.request, json, struct, zlib

r = urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=5)
print("HEALTH:", json.loads(r.read()))

width, height = 128, 128
raw = bytearray()
for y in range(height):
    raw.append(0)
    for x in range(width):
        raw.append(x % 256)

def make_png(w, h, raw_data):
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    idat = zlib.compress(bytes(raw_data))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

png_data = make_png(width, height, raw)

boundary = "----TestBoundary"
body = []
body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"cloud.png\"\r\nContent-Type: image/png\r\n\r\n".encode() + png_data)
body.append(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"temperature\"\r\n\r\n25".encode())
body.append(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"humidity\"\r\n\r\n70".encode())
body.append(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"wind_speed\"\r\n\r\n12".encode())
body.append(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"pressure\"\r\n\r\n1005".encode())
body.append(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"district\"\r\n\r\nDhaka".encode())
body.append(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"hour\"\r\n\r\n14".encode())
body.append(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"month\"\r\n\r\n7".encode())
body.append(f"\r\n--{boundary}--\r\n".encode())
data = b"".join(body)

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/predict",
    data=data,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
r = urllib.request.urlopen(req, timeout=15)
print("PREDICT:", json.loads(r.read()))

req2 = urllib.request.Request("http://127.0.0.1:8000/api/report")
r2 = urllib.request.urlopen(req2, timeout=15)
print("REPORT: downloaded", len(r2.read()), "bytes")
