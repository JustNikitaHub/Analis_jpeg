import numpy as np
from typing import List, Tuple
import os
import struct

Luminance_DC_differences= [
	"00",
	"010",
	"011",
	"100",
	"101",
	"110",
	"1110",
	"11110",
	"111110",
	"1111110",
	"11111110",
	"111111110",
]

Luminance_AC = [
			[
"1010",
"00",
"01",
"100",
"1011",
"11010",
"1111000",
"11111000",
"1111110110",
"1111111110000010",
"1111111110000011"
            ],
			[
"",
"1100",
"11011",
"1111001",
"111110110",
"11111110110",
"1111111110000100",
"1111111110000101",
"1111111110000110",
"1111111110000111",
"1111111110001000"
            ],
			[
"",
"11100",
"11111001",
"1111110111",
"111111110100",
"1111111110001001",
"1111111110001010",
"1111111110001011",
"1111111110001100",
"1111111110001101",
"1111111110001110"
            ],
			[
"",
"111010",
"111110111",
"111111110101",
"1111111110001111",
"1111111110010000",
"1111111110010001",
"1111111110010010",
"1111111110010011",
"1111111110010100",
"1111111110010101"
            ],
			[
"",
"111011",
"1111111000",
"1111111110010110",
"1111111110010111",
"1111111110011000",
"1111111110011001",
"1111111110011010",
"1111111110011011",
"1111111110011100",
"1111111110011101"
            ],
			[
"",
"1111010",
"11111110111",
"1111111110011110",
"1111111110011111",
"1111111110100000",
"1111111110100001",
"1111111110100010",
"1111111110100011",
"1111111110100100",
"1111111110100101"
            ],
			[
"",
"1111011",
"111111110110",
"1111111110100110",
"1111111110100111",
"1111111110101000",
"1111111110101001",
"1111111110101010",
"1111111110101011",
"1111111110101100",
"1111111110101101"
            ],
			[
"",
"11111010",
"111111110111",
"1111111110101110",
"1111111110101111",
"1111111110110000",
"1111111110110001",
"1111111110110010",
"1111111110110011",
"1111111110110100",
"1111111110110101"
            ],
			[
"",
"111111000",
"111111111000000",
"1111111110110110",
"1111111110110111",
"1111111110111000",
"1111111110111001",
"1111111110111010",
"1111111110111011",
"1111111110111100",
"1111111110111101"
            ],
			[
"",
"111111001",
"1111111110111110",
"1111111110111111",
"1111111111000000",
"1111111111000001",
"1111111111000010",
"1111111111000011",
"1111111111000100",
"1111111111000101",
"1111111111000110"
            ],
			[
"",
"111111010",
"1111111111000111",
"1111111111001000",
"1111111111001001",
"1111111111001010",
"1111111111001011",
"1111111111001100",
"1111111111001101",
"1111111111001110",
"1111111111001111"
            ],
			[
"",
"1111111001",
"1111111111010000",
"1111111111010001",
"1111111111010010",
"1111111111010011",
"1111111111010100",
"1111111111010101",
"1111111111010110",
"1111111111010111",
"1111111111011000"
            ],
			[
"",
"1111111010",
"1111111111011001",
"1111111111011010",
"1111111111011011",
"1111111111011100",
"1111111111011101",
"1111111111011110",
"1111111111011111",
"1111111111100000",
"1111111111100001"
            ],
			[
"",
"11111111000",
"1111111111100010",
"1111111111100011",
"1111111111100100",
"1111111111100101",
"1111111111100110",
"1111111111100111",
"1111111111101000",
"1111111111101001",
"1111111111101010"
            ],
			[
"",
"1111111111101011",
"1111111111101100",
"1111111111101101",
"1111111111101110",
"1111111111101111",
"1111111111110000",
"1111111111110001",
"1111111111110010",
"1111111111110011",
"1111111111110100"
            ],
			[
"11111111001",
"1111111111110101",
"1111111111110110",
"1111111111110111",
"1111111111111000",
"1111111111111001",
"1111111111111010",
"1111111111111011",
"1111111111111100",
"1111111111111101",
"1111111111111110"
            ]]

Chrominance_DC_differences = [
	"00",
	"01",
	"10",
	"110",
	"1110",
	"11110",
	"111110",
	"1111110",
	"11111110",
	"111111110",
	"1111111110",
	"11111111110",
]

Chrominance_AC = [
			[
"00",
"01",
"100",
"1010",
"11000",
"11001",
"111000",
"1111000",
"111110100",
"1111110110",
"111111110100"
            ],
			[
"",
"1011",
"111001",
"11110110",
"111110101",
"11111110110",
"111111110101",
"1111111110001000",
"1111111110001001",
"1111111110001010",
"1111111110001011"
			],
			[
"",
"11010",
"11110111",
"1111110111",
"111111110110",
"111111111000010",
"1111111110001100",
"1111111110001101",
"1111111110001110",
"1111111110001111",
"1111111110010000"
			],
			[
"",
"11011",
"11111000",
"1111111000",
"111111110111",
"1111111110010001",
"1111111110010010",
"1111111110010011",
"1111111110010100",
"1111111110010101",
"1111111110010110"
			],
			[
"",
"111010",
"111110110",
"1111111110010111",
"1111111110011000",
"1111111110011001",
"1111111110011010",
"1111111110011011",
"1111111110011100",
"1111111110011101",
"1111111110011110"
			],
			[
"",
"111011",
"1111111001",
"1111111110011111",
"1111111110100000",
"1111111110100001",
"1111111110100010",
"1111111110100011",
"1111111110100100",
"1111111110100101",
"1111111110100110"
			],
			[
"",
"1111001",
"11111110111",
"1111111110100111",
"1111111110101000",
"1111111110101001",
"1111111110101010",
"1111111110101011",
"1111111110101100",
"1111111110101101",
"1111111110101110"
			],
			[
"",
"1111010",
"11111111000",
"1111111110101111",
"1111111110110000",
"1111111110110001",
"1111111110110010",
"1111111110110011",
"1111111110110100",
"1111111110110101",
"1111111110110110"
			],
			[
"",
"11111001",
"1111111110110111",
"1111111110111000",
"1111111110111001",
"1111111110111010",
"1111111110111011",
"1111111110111100",
"1111111110111101",
"1111111110111110",
"1111111110111111"
			],
			[
"",
"111110111",
"1111111111000000",
"1111111111000001",
"1111111111000010",
"1111111111000011",
"1111111111000100",
"1111111111000101",
"1111111111000110",
"1111111111000111",
"1111111111001000"
			],
			[
"",
"111111000",
"1111111111001001",
"1111111111001010",
"1111111111001011",
"1111111111001100",
"1111111111001101",
"1111111111001110",
"1111111111001111",
"1111111111010000",
"1111111111010001"
			],
			[
"",
"111111001",
"1111111111010010",
"1111111111010011",
"1111111111010100",
"1111111111010101",
"1111111111010110",
"1111111111010111",
"1111111111011000",
"1111111111011001",
"1111111111011010"
			],
			[
"",
"111111010",
"1111111111011011",
"1111111111011100",
"1111111111011101",
"1111111111011110",
"1111111111011111",
"1111111111100000",
"1111111111100001",
"1111111111100010",
"1111111111100011"
			],
			[
"",
"11111111001",
"1111111111100100",
"1111111111100101",
"1111111111100110",
"1111111111100111",
"1111111111101000",
"1111111111101001",
"1111111111101010",
"1111111111101011",
"1111111111101100"
			],
			[
"",
"11111111100000",
"1111111111101101",
"1111111111101110",
"1111111111101111",
"1111111111110000",
"1111111111110001",
"1111111111110010",
"1111111111110011",
"1111111111110100",
"1111111111110101"
			],
			[
"1111111010",
"111111111000011",
"1111111111110110",
"1111111111110111",
"1111111111111000",
"1111111111111001",
"1111111111111010",
"1111111111111011",
"1111111111111100",
"1111111111111101",
"1111111111111110"
			]
]

# 1. Преобразование RGB в YCbCr
def rgb_to_ycbcr(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    
    # Обеспечиваем, чтобы значения оставались в диапазоне [0, 255]
    Y = np.clip(Y, 0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)
    
    return Y, Cb, Cr

# 2. Даунсэмплинг 4:2:0 (применяется к Cb и Cr)
def downsample(c: np.ndarray) -> np.ndarray:
    height, width = c.shape
    # Убедимся, что размеры четные
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Image dimensions must be even for 4:2:0 subsampling")
    
    # Используем усреднение для даунсэмплинга
    downsampled = (c[::2, ::2].astype(np.uint16) + 
                   c[::2, 1::2] + 
                   c[1::2, ::2] + 
                   c[1::2, 1::2]) // 4
    
    return downsampled.astype(np.uint8)

def pad_to_multiple(img: np.ndarray, multiple: int = 16) -> np.ndarray:
    """Дополняет изображение нулями до нужных размеров"""
    h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')

# 3. Разбиение изображения на блоки 8x8
def split_into_8x8_blocks(Ycbcr: np.ndarray) -> List[np.ndarray]:
    if Ycbcr.size == 0:
        raise ValueError("Input array is empty")
    
    height, width = Ycbcr.shape
    
    # Проверяем, что размеры кратны 8
    if height % 8 != 0 or width % 8 != 0:
        if height%8!=0: print(f"Image dimensions (height={height}) must be multiples of 8")
        if width%8!=0: print(f"Image dimensions (width={width}) must be multiples of 8")
        raise ValueError
    
    blocks = []
    
    # Проходим по изображению с шагом 8x8
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = Ycbcr[i:i+8, j:j+8]
            blocks.append(block)
    
    return blocks

# 4.1 Прямое DCT-II преобразование для блока 8x8
def dct_2d_8x8(block: np.ndarray) -> np.ndarray:
    N = 8
    PI = np.pi
    SQRT2 = np.sqrt(2)
    
    # Коэффициенты масштабирования
    def C(u):
        return 1/SQRT2 if u == 0 else 1.0
    
    # Создаем матрицу коэффициентов DCT
    dct_matrix = np.zeros((N, N))
    
    for u in range(N):
        for v in range(N):
            # Вычисляем коэффициент для (u,v)
            sum_val = 0.0
            for x in range(N):
                for y in range(N):
                    sum_val += block[y, x] * np.cos((2*x + 1) * u * PI / (2*N)) * \
                                             np.cos((2*y + 1) * v * PI / (2*N))
            
            # Применяем нормализацию
            dct_matrix[v, u] = 0.25 * C(u) * C(v) * sum_val
    
    return dct_matrix

# Оптимизированная версия DCT с предварительно вычисленными таблицами косинусов
def precompute_cos_table():
    N = 8
    PI = np.pi
    cos_table = np.zeros((N, N))
    
    for u in range(N):
        for x in range(N):
            cos_table[u, x] = np.cos((2 * x + 1) * u * PI / (2 * N))
    
    return cos_table

COS_TABLE = precompute_cos_table()

def dct_2d_8x8_optimized(block: np.ndarray) -> np.ndarray:
    N = 8
    SQRT2 = np.sqrt(2)
    
    def C(u):
        return 1/SQRT2 if u == 0 else 1.0
    
    # Горизонтальное DCT
    temp = np.zeros((N, N))
    for y in range(N):
        for u in range(N):
            sum_val = 0.0
            for x in range(N):
                sum_val += block[y, x] * COS_TABLE[u, x]
            temp[y, u] = sum_val * C(u) * 0.5
    
    # Вертикальное DCT
    coeffs = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            sum_val = 0.0
            for y in range(N):
                sum_val += temp[y, u] * COS_TABLE[v, y]
            coeffs[v, u] = sum_val * C(v) * 0.5
    
    return coeffs

# Константы для DCT
N = 8
PI = np.pi
SQRT2 = np.sqrt(2)
C0 = 1.0 / SQRT2
C1 = 1.0

# 4.2 Обратное DCT-II преобразование для блока NxN
def idct_1d(input_arr: np.ndarray) -> np.ndarray:
    output = np.zeros(N)
    for x in range(N):
        sum_val = 0.0
        for u in range(N):
            cu = C0 if u == 0 else C1
            sum_val += cu * input_arr[u] * COS_TABLE[u, x]
        output[x] = sum_val * 0.5  # Нормализация
    return output

def idct_2d_8x8(coeffs: np.ndarray) -> np.ndarray:
    temp = np.zeros((N, N))
    block = np.zeros((N, N))
    
    # Применяем 1D IDCT к каждому столбцу (вертикальное преобразование)
    for u in range(N):
        column = coeffs[:, u]
        idct_column = idct_1d(column)
        temp[:, u] = idct_column
    
    # Применяем 1D IDCT к каждой строке (горизонтальное преобразование)
    for y in range(N):
        row = temp[y, :]
        idct_row = idct_1d(row)
        block[y, :] = idct_row
    
    return block

# 5. Генерация матрицы квантования для заданного уровня качества
# Стандартная матрица квантования Y для качества 50
Luminance_quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.int16)

# Стандартная матрица квантования Cb и Cr для качества 50
Chrominance_quantization_table = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.int16)

def generate_quantization_matrix(quality: int, quantization_table: np.ndarray) -> np.ndarray:
    # Корректируем качество (1-100)
    quality = max(1, min(100, quality))
    
    # Вычисляем scale_factor
    if quality < 50:
        scale_factor = 50.0 / quality  # Для quality=1 -> scale_factor=50
    else:
        scale_factor = 2.0 - (quality / 50.0)  # Для quality=100 -> scale_factor=0
    
    # Масштабируем стандартную матрицу
    q_matrix = np.round(quantization_table * scale_factor).astype(np.int16)
    q_matrix = np.maximum(q_matrix, 1)  # Значения должны быть не меньше 1
    
    return q_matrix

# 6.1 Квантование DCT коэффициентов
def quantize(dct_coeffs: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """Квантование DCT коэффициентов с использованием матрицы квантования"""
    quantized = np.round(dct_coeffs / q_matrix).astype(np.int16)
    return quantized

# 6.2 Обратное квантование (восстановление DCT-коэффициентов)
def dequantize(quantized: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """Обратное квантование коэффициентов"""
    dct_coeffs = quantized.astype(np.float64) * q_matrix
    return dct_coeffs

# 7.1 Зигзаг-сканирование блока
zigzag_order = np.array([
    0,
    1, 8,
    16, 9, 2,
    3, 10, 17, 24,
    32, 25, 18, 11, 4,
    5, 12, 19, 26, 33, 40,
    48, 41, 34, 27, 20, 13, 6,
    7, 14, 21, 28, 35, 42, 49, 56,
    57, 50, 43, 36, 29, 22, 15,
    23, 30, 37, 44, 51, 58,
    59, 52, 45, 38, 31,
    39, 46, 53, 60,
    61, 54, 47,
    55, 62,
    63
], dtype=np.int32)

def zigzag_scan_fast(quantized: np.ndarray) -> np.ndarray:
    """Безопасное зигзаг-сканирование блока 8x8 с проверками"""
    if quantized.shape != (8, 8):
        raise ValueError(f"Ожидается блок 8x8, получен {quantized.shape}")
    
    zigzag = np.zeros(64, dtype=np.int16)
    
    for i in range(64):
        idx = zigzag_order[i]
        if idx >= 64:
            raise ValueError(f"Неверный индекс {idx} в zigzag_order")
        
        row, col = idx // 8, idx % 8
        zigzag[i] = quantized[row, col]
    
    return zigzag

def create_jpeg_header(width, height, q_y, q_c):
    # Маркер начала изображения
    soi = b"\xFF\xD8"
    
    # Заголовок APP0 (минимально необходимый)
    app0 = (b"\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    
    # Таблицы квантования (пример для качества 50)
    dqt = (b"\xFF\xDB\x00\x84\x00" + q_y.tobytes() + 
           b"\xFF\xDB\x00\x84\x01" + q_c.tobytes())
    
    # Заголовок кадра (SOF0)
    sof0 = (b"\xFF\xC0\x00\x11\x08" + 
            height.to_bytes(2, 'big') + 
            width.to_bytes(2, 'big') + 
            b"\x03\x11\x00\x12\x00\x13\x00")
    
    return soi + app0 + dqt + sof0

# 7.2 Обратное зигзаг-сканирование блока
def inverse_zigzag_scan(zigzag_coeffs: np.ndarray) -> np.ndarray:
    """Восстановление блока 8x8 из зигзаг-последовательности"""
    block = np.zeros((8, 8), dtype=np.int16)
    for i in range(64):
        idx = zigzag_order[i]
        block[idx // 8][idx % 8] = zigzag_coeffs[i]
    return block

# 8.1 Разностное кодирование DC коэффициентов
def dc_difference(coeffs: np.ndarray) -> None:
    """Разностное кодирование DC коэффициентов"""
    if len(coeffs) == 0:
        return
    
    # Обрабатываем каждый блок (предполагается, что coeffs - плоский массив)
    for i in range(64, len(coeffs), 64):
        coeffs[i] -= coeffs[i-64]

def calculate_dc_category(dc_value: int) -> int:
    """Вычисляет категорию DC-коэффициента (0-11)"""
    if dc_value == 0:
        return 0
    abs_value = abs(dc_value)
    return min(len(bin(abs_value)) - 2, 11)  # Ограничиваем максимум 11

def calculate_ac_category(value: int) -> int:
    if value == 0:
        return 0
    return min(len(bin(abs(value))) - 2, 10)

def calculate_ac_size(value: int) -> int:
    if value == 0:
        return 0
    abs_val = abs(value)
    return min(len(bin(abs_val)) - 2, 10)

# 9. Переменное кодирование разностей DC и AC коэффициентов
def int_to_binary_vector(num: int, positive1_or_negative0: int = 1) -> List[int]:
    """Преобразование числа в бинарный вектор"""
    bits = []
    
    if num == 0:
        bits.append(0)
        return bits
    
    if positive1_or_negative0 == 0:
        num *= -1
    
    # Разложение числа на биты
    while num > 0:
        bits.append(1 if num % 2 == positive1_or_negative0 else 0)  # Младший бит
        num = num // 2
    
    # Разворачиваем биты (старший бит первым)
    bits.reverse()
    
    return bits

def encode_huffman_data(prepared_data: List[Tuple[int, List[int], List[Tuple[int, int]]]], dc_table: List[str], ac_table: List[List[str]]) -> str:
    encoded_bits = ""
    for category, dc_bits, ac_pairs in prepared_data:
        # Кодируем DC
        safe_category = min(category, len(dc_table)-1)
        encoded_bits += dc_table[safe_category]
        encoded_bits += "".join(map(str, dc_bits))
        
        # Кодируем AC
        for run, size in ac_pairs:
            safe_run = min(run, len(ac_table)-1)
            safe_size = min(size, len(ac_table[0])-1)
            encoded_bits += ac_table[safe_run][safe_size]
    return encoded_bits

def rle_encode_ac(num: int, rle_str: List[int], zero_count: int, EOB: bool) -> bool:
    """RLE кодирование AC коэффициентов"""
    if num == 0:
        zero_count += 1
        if zero_count > 15:
            rle_str.append(15)
            rle_str.append(0)
            zero_count = 0
            EOB = True
        return True
    else:
        rle_str.append(zero_count)
        return False

def prepare_for_huffman(zigzag_data: List[int], is_luminance: bool) -> List[Tuple[int, List[int], List[Tuple[int, int]]]]:
    prepared = []
    for i in range(0, len(zigzag_data), 64):
        # DC коэффициент
        dc_value = zigzag_data[i]
        category = calculate_dc_category(dc_value)
        
        # Битовая последовательность значения DC
        dc_bits = []
        if category > 0:
            if dc_value > 0:
                dc_bits = [int(b) for b in bin(dc_value)[2:].zfill(category)]
            else:
                dc_bits = [int(b) for b in bin(abs(dc_value) ^ ((1 << category) - 1))[2:]]
        
        # AC коэффициенты
        ac_pairs = []
        zero_count = 0
        for j in range(i+1, min(i+64, len(zigzag_data))):
            if zigzag_data[j] == 0:
                zero_count += 1
                if zero_count == 16:
                    ac_pairs.append((15, 0))
                    zero_count = 0
            else:
                value = zigzag_data[j]
                size = calculate_ac_size(value)
                ac_pairs.append((zero_count, size))
                zero_count = 0
        
        # Добавляем маркер EOB если нужно
        if zero_count > 0:
            ac_pairs.append((0, 0))
        
        prepared.append((category, dc_bits, ac_pairs))
    return prepared

# Подготовка DC и AC коэффициентов для кодирования
def preparing_for_coding_dc_and_ac(coefficients: List[int]) -> List[int]:
    """Подготовка DC и AC коэффициентов для кодирования"""
    output = []
    
    # Разностное кодирование DC коэффициентов
    dc_difference(coefficients)
    
    size = len(coefficients)
    for i in range(0, size, 64):  # Обработка каждого блока 8x8 (64 коэффициента)
        temp = []
        if i >= size:
            print("в prep")
            break
        
        # Обработка DC коэффициента
        dc_coeff = coefficients[i]
        if dc_coeff >= 0:
            temp = int_to_binary_vector(dc_coeff, 1)
        else:
            temp = int_to_binary_vector(dc_coeff, 0)
        
        output.append(len(temp))  # Категория (длина битового представления)
        output.extend(temp)      # Сами биты
        
        # Обработка AC коэффициентов
        zero_count = 0
        EOB = False
        for j in range(i + 1, i + 64):  # Остальные 63 коэффициента в блоке
            if j >= len(coefficients):
                break
                
            EOB = False
            if rle_encode_ac(coefficients[j], output, zero_count, EOB):
                continue
            
            # Если не ноль, кодируем значение
            ac_coeff = coefficients[j]
            if ac_coeff >= 0:
                temp = int_to_binary_vector(ac_coeff, 1)
            else:
                temp = int_to_binary_vector(ac_coeff, 0)
            
            output.append(len(temp))  # Категория
            output.extend(temp)      # Сами биты
            zero_count = 0
        
        # Добавляем маркер конца блока (0,0) если нужно
        if zero_count != 0:
            output.append(0)
            output.append(0)
        
        if EOB:  # Случай, когда было ровно 16 нулей до конца блока
            if len(output) >= 2:
                output[-2] = 0
    
    return output

def HA_encode(data: List[int], DC_differences: List[str], AC: List[List[str]]) -> str:
    """Кодирование данных с использованием таблиц Хаффмана для DC и AC коэффициентов"""
    encoded = ""
    i = 0
    size = len(data)
    
    while i < size:
        # 1. Кодирование DC коэффициента
        if i >= size:
            break  # Защита от выхода за границы
            
        dc_category = data[i]
        if dc_category >= len(DC_differences):
            raise ValueError(f"Неверная категория DC: {dc_category}")
        encoded += DC_differences[dc_category]
        i += 1
        
        # 2. Добавление битов значения DC
        k_size = dc_category
        for _ in range(k_size):
            if i >= size:
                break
            encoded += str(data[i])
            i += 1

        # 3. Кодирование AC коэффициентов
        count = 1  # Уже обработали DC
        
        while count < 64 and i < size:
            # Проверка маркера конца блока (0/0)
            if i+1 < size and data[i] == 0 and data[i+1] == 0:
                encoded += AC[0][0]
                i += 2
                break
            
            # Проверка на 16 нулей (15/0)
            if i+1 < size and data[i] == 15 and data[i+1] == 0:
                if 15 < len(AC) and 0 < len(AC[15]):
                    encoded += AC[15][0]
                count += 16
                i += 2
                continue
            
            # Обработка обычных AC коэффициентов
            if i+1 >= size:
                break
                
            run = data[i]
            size_ac = data[i+1]
            
            # Проверка границ таблицы AC
            if run >= len(AC) or size_ac >= len(AC[run]):
                raise ValueError(f"Неверная пара (run,size)=({run},{size_ac})")
                
            encoded += AC[run][size_ac]
            i += 2
            
            # Добавление битов значения AC
            for _ in range(size_ac):
                if i >= size:
                    break
                encoded += str(data[i])
                i += 1
            
            count += run + 1
    
    return encoded

def pack_bits_to_bytes(bit_string: str) -> bytes:
    """Упаковка битовой строки в байты с указанием количества дополняющих битов"""
    output = bytearray()
    len_bit_string = len(bit_string)
    
    # Вычисляем количество дополняющих битов
    padding_bits = (8 - len_bit_string % 8) % 8
    # Дополняем строку нулями
    padded_string = bit_string + '0' * padding_bits
    
    # Первый байт - количество дополняющих битов
    output.append(padding_bits)
    
    # Упаковываем по 8 бит в байты
    for i in range(0, len(padded_string), 8):
        byte_str = padded_string[i:i+8]
        byte = int(byte_str, 2)
        output.append(byte)
    
    return bytes(output)

def rle_decode_ac(rle_encoded: List[Tuple[int, int]], total_ac: int) -> List[int]:
    """Обратное RLE кодирование AC коэффициентов"""
    ac_coeffs = []
    
    for run, value in rle_encoded:
        if run == 0 and value == 0:
            # Маркер конца блока - заполняем оставшиеся нулями
            ac_coeffs.extend([0] * (total_ac - len(ac_coeffs)))
            break
        
        # Добавляем нули
        ac_coeffs.extend([0] * run)
        
        # Добавляем ненулевое значение
        ac_coeffs.append(value)
    
    return ac_coeffs

def reverse_difference_dc(delta_encoded: List[int]) -> List[int]:
    """Обратное разностное кодирование DC коэффициентов"""
    if not delta_encoded:
        return []
    
    dc_coeffs = [delta_encoded[0]]
    
    for i in range(1, len(delta_encoded)):
        # Восстанавливаем оригинальное значение DC
        dc_coeffs.append(dc_coeffs[i-1] + delta_encoded[i])
    
    return dc_coeffs

def main():
    try:
        # Параметры изображения
        filename = "color.raw"  # 800x600 (width x height)
        width = 800
        height = 600

        # Чтение файла
        with open(filename, 'rb') as ifT:
            data = ifT.read()
            size = len(data)
            
            if size != 3 * height * width:
                print(f"Ошибка: разрешение не соответствует размеру файла: {size}")
                return 0
            
            print(f"Параметры изображения:\n1) Размер: {size} байт\n")
            print(f"2) Разрешение: {height}x{width}\n")

            # Декодирование RGB данных
            r = np.zeros((height, width), dtype=np.uint8)
            g = np.zeros((height, width), dtype=np.uint8)
            b = np.zeros((height, width), dtype=np.uint8)

            for y in range(height):
                for x in range(width):
                    offset = (y * width + x) * 3
                    r[y, x] = data[offset]
                    g[y, x] = data[offset + 1]
                    b[y, x] = data[offset + 2]

        # Преобразование RGB в YCbCr
        Y, Cb, Cr = rgb_to_ycbcr(r, g, b)

        # Даунсэмплинг 4:2:0
        if height % 4 != 0 or width % 4 != 0:
            print("Ошибка: разрешение не делится на 4")
            return 0

        Cb_down = downsample(Cb)
        Cr_down = downsample(Cr)

        Y = pad_to_multiple(Y)
        Cb_down = pad_to_multiple(Cb_down)
        Cr_down = pad_to_multiple(Cr_down)

        # Разбиение на блоки 8x8
        print(f"{Y} - Y CHECK")
        Y_blocks = split_into_8x8_blocks(Y)
        print(f"{Cb} - Cb CHECK")
        Cb_blocks = split_into_8x8_blocks(Cb_down)
        print(f"{Cr} - Cr CHECK")
        Cr_blocks = split_into_8x8_blocks(Cr_down)

        # Генерация матриц квантования
        quality = 50
        qY = generate_quantization_matrix(quality, Luminance_quantization_table)
        qC = generate_quantization_matrix(quality, Chrominance_quantization_table)
        print("после квантов")

        # DCT и квантование
        Y_quantized = []
        for block in Y_blocks:
            dct = dct_2d_8x8_optimized(block)
            quantized = quantize(dct, qY)
            Y_quantized.append(quantized)
        print("после Y_q")
        Cb_quantized = []
        Cr_quantized = []
        for cb_block, cr_block in zip(Cb_blocks, Cr_blocks):
            cb_dct = dct_2d_8x8_optimized(cb_block)
            cr_dct = dct_2d_8x8_optimized(cr_block)
            Cb_quantized.append(quantize(cb_dct, qC))
            Cr_quantized.append(quantize(cr_dct, qC))
        print("после CbCr_q")
        print("Проверка Y_quantized:")
        print("Размер:", len(Y_quantized), "блоков")
        print("Максимальное значение в блоках:", max(np.max(block) for block in Y_quantized))
        print("Минимальное значение в блоках:", min(np.min(block) for block in Y_quantized))

        # Зигзаг-сканирование
        Y_zigzag = []
        for block in Y_quantized:
            Y_zigzag.extend(zigzag_scan_fast(block))
        print("обход Z Y")

        Cb_zigzag = []
        for block in Cb_quantized:
            Cb_zigzag.extend(zigzag_scan_fast(block))
        print("обход Z Cb")

        Cr_zigzag = []
        for block in Cr_quantized:
            Cr_zigzag.extend(zigzag_scan_fast(block))
        print("обход Z Cr")

        # Подготовка к кодированию (разностное кодирование DC)
        preparing_for_coding_dc_and_ac(Y_zigzag)
        preparing_for_coding_dc_and_ac(Cb_zigzag)
        preparing_for_coding_dc_and_ac(Cr_zigzag)
        print("после prep")
        # 1. Подготовка данных для Хаффмана
        y_prepared = prepare_for_huffman(Y_zigzag, is_luminance=True)
        cb_prepared = prepare_for_huffman(Cb_zigzag, is_luminance=False)
        cr_prepared = prepare_for_huffman(Cr_zigzag, is_luminance=False)

        # 2. Кодирование каждой компоненты
        y_encoded = encode_huffman_data(
            y_prepared, 
            Luminance_DC_differences, 
            Luminance_AC
        )
        cb_encoded = encode_huffman_data(
            cb_prepared, 
            Chrominance_DC_differences, 
            Chrominance_AC
        )
        cr_encoded = encode_huffman_data(
            cr_prepared, 
            Chrominance_DC_differences, 
            Chrominance_AC
        )

        # 3. Объединение всех битовых строк
        full_encoded = y_encoded + cb_encoded + cr_encoded
        print("Закодированные биты Y/Cb/Cr готовы")

        # Упаковка в байты
        jpeg_data = pack_bits_to_bytes(full_encoded)

        # Вывод результата
        print("JPEG данные:")
        print(jpeg_data[:100])  # Печатаем первые 100 байт для примера
        with open('output.jpg', 'wb') as f:
            f.write(jpeg_data)

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    main()