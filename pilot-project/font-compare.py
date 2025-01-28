from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm
import math
import json
from collections import Counter

def calculate_segment_length(p1, p2):
    """두 점 사이의 거리를 계산합니다."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_segment_angle(p1, p2):
    """두 점 사이의 각도를 계산합니다."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)

def normalize_contour(contour):
    """윤곽선 좌표를 정규화합니다."""
    if not contour:
        return []
    min_x = min(p[0] for p in contour)
    min_y = min(p[1] for p in contour)
    max_x = max(p[0] for p in contour)
    max_y = max(p[1] for p in contour)
    if max_x == min_x or max_y == min_y:  # 크기가 0인 경우 방지
        return []
    return [( (x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y) ) for x, y in contour]

class ContourExtractor(BasePen):
    def __init__(self, glyphSet):
        BasePen.__init__(self, glyphSet)
        self.contours = []
        self.currentContour = []

    def moveTo(self, p0):
        if self.currentContour:
            self.contours.append(self.currentContour)
        self.currentContour = [p0]

    def lineTo(self, p1):
        self.currentContour.append(p1)

    def curveTo(self, *points):
        self.currentContour.extend(points)

    def qCurveTo(self, *points):
        self.currentContour.extend(points)

    def closePath(self):
        self.contours.append(self.currentContour)
        self.currentContour = []

    def endPath(self):
        if self.currentContour:
            self.contours.append(self.currentContour)
            self.currentContour = []

def glyph_to_vector(font_path, unicode_val): # unicode_val로 변경
    """폰트 파일에서 특정 유니코드 값에 해당하는 글리프의 윤곽선 데이터를 벡터로 변환합니다."""
    try:
        font = TTFont(font_path)
        glyphSet = font.getGlyphSet()
        cmap = font.getBestCmap()
        glyph_name = None
        if cmap and unicode_val in cmap:
            glyph_name = cmap[unicode_val]
        if not glyph_name or glyph_name not in glyphSet:
            return None

        glyph = glyphSet[glyph_name]
        pen = ContourExtractor(glyphSet)
        glyph.draw(pen)
        contours = pen.contours

        if not contours:
            return None

        contour_features = []
        for contour in contours:
            normalized_contour = normalize_contour(contour)
            if not normalized_contour:
                continue
            for i in range(len(normalized_contour)):
                p1 = normalized_contour[i]
                p2 = normalized_contour[(i + 1) % len(normalized_contour)]
                length = calculate_segment_length(p1, p2)
                angle = calculate_segment_angle(p1, p2)
                contour_features.extend([length, angle])

        if not contour_features:
            return None
        return np.array(contour_features).tolist()

    except Exception as e:
        print(f"Error processing glyph U+{unicode_val:04X} in {font_path}: {e}") # 오류 메시지 수정
        return None

def load_font_vectors(json_path):
    """JSON 파일에서 폰트 벡터 데이터를 로드합니다."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            font_vectors = json.load(f)
            return font_vectors
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return None

def pad_vector(vector, target_length):
    """벡터를 특정 길이로 패딩합니다."""
    if vector is None:
        return np.zeros(target_length).tolist()
    current_length = len(vector)
    if current_length < target_length:
        padding = [0] * (target_length - current_length)
        return vector + padding
    elif current_length > target_length:
        return vector[:target_length]
    return vector

def get_korean_syllable_type(char):
    """한글 음절의 유형(초성, 중성, 종성)을 반환합니다."""
    if '가' <= char <= '힣':
        code = ord(char) - ord('가')
        jong = code % 28
        jung = (code // 28) % 21
        cho = code // (28 * 21)
        if jong == 0:
            return "초성중성"
        else:
            return "초성중성종성"
    return None

def harmonic_mean(numbers):
    """숫자들의 조화 평균을 계산합니다. 0이 포함된 경우 0을 반환합니다."""
    numbers = [num for num in numbers if num > 0] # 0 제거
    if not numbers:
        return 0
    return len(numbers) / sum(1 / num for num in numbers)

def compare_font_with_json(json_path, target_font_path, chars):
    """JSON 데이터와 타겟 폰트를 비교합니다."""
    loaded_vectors = load_font_vectors(json_path)
    if loaded_vectors is None:
        return None

    group_similarities = {"초성중성": [], "초성중성종성": []}

    max_vector_length = 0
    for char in chars:
        unicode_val = ord(char)
        target_vector = glyph_to_vector(target_font_path, unicode_val)
        if target_vector is not None:
            max_vector_length = max(max_vector_length, len(target_vector))

    for font_name, vectors in loaded_vectors.items():
        for char, vector in vectors.items():
            if vector is not None:
                max_vector_length = max(max_vector_length, len(vector))

    for char in tqdm(chars, desc="Comparing Characters"):
        unicode_val = ord(char)
        target_vector = glyph_to_vector(target_font_path, unicode_val)
        if target_vector is None:
            continue
        target_vector = pad_vector(target_vector, max_vector_length)

        syllable_type = get_korean_syllable_type(char)
        if syllable_type is None:
            continue

        char_similarities = []
        for font_name, vectors in loaded_vectors.items():
            if char in vectors:
                vector = vectors[char]
                vector = pad_vector(vector, max_vector_length)
                if np.all(np.array(vector) == 0):
                    char_similarities.append(0)  # 0으로 처리
                    continue
                try:
                    similarity = cosine_similarity([target_vector], [vector])[0, 0]
                    char_similarities.append(similarity)
                except ValueError as e:
                    print(f"ValueError: {e}, char: {char}")
                    char_similarities.append(0) # 오류 발생 시 0으로 처리
                    continue
            else:
                char_similarities.append(0) # 벡터가 없는 경우 0으로 처리

        group_similarities[syllable_type].append(char_similarities)

    final_similarities = []
    font_names = list(loaded_vectors.keys())
    num_fonts = len(font_names)

    for i in range(num_fonts):
        group_sims = []
        for group, sims in group_similarities.items():
            valid_sims = [sim[i] for sim in sims if len(sim) > i and sim[i] != 0] # -1 이 아닌 0으로 변경
            if valid_sims:
                group_sims.append(harmonic_mean(valid_sims))
            else:
                group_sims.append(0)

        weights = {"초성중성": 0.6, "초성중성종성": 0.4}
        final_sim = 0
        total_weight = 0
        for group, sim in zip(group_similarities.keys(), group_sims):
            final_sim += sim * weights[group]
            total_weight += weights[group]
        if total_weight > 0:
            final_sim /= total_weight
        else:
            final_sim = -1

        final_similarities.append(final_sim)

    font_similarity_pairs = []
    for i, sim in enumerate(final_similarities):
        if sim != -1:
            font_similarity_pairs.append((font_names[i], sim))

    sorted_similarities = sorted(font_similarity_pairs, key=lambda x: x[1], reverse=True)

    return sorted_similarities

def recommend_top_3_fonts(json_path, target_font_path, chars):
    sorted_similarities = compare_font_with_json(json_path, target_font_path, chars)
    if sorted_similarities is None:
        return
    
    if not sorted_similarities:
        print("No similar fonts found.")
        return

    print(f"'{os.path.basename(target_font_path)}'와 유사한 폰트 TOP 3:")
    for i, (font_name, similarity) in enumerate(sorted_similarities[:5]):
        print(f"{i+1}. {font_name} (유사도: {similarity:.4f})")

def vectorize_fonts(font_folder, output_json, chars):
    """폰트 폴더 내의 모든 폰트를 벡터화하여 JSON 파일로 저장합니다."""
    font_paths = [os.path.join(font_folder, f) for f in os.listdir(font_folder) if f.lower().endswith(('.ttf', '.otf'))]
    font_vectors = {}

    for font_path in tqdm(font_paths, desc="Vectorizing Fonts"):
        font_name = os.path.basename(font_path)
        font_vectors[font_name] = {}
        try:
            for char in chars:
                unicode_val = ord(char)
                vector = glyph_to_vector(font_path, unicode_val) # 유니코드 값 전달
                if vector is not None:
                    font_vectors[font_name][char] = vector
                else:
                    print(f"Warning: Could not vectorize '{char}' (U+{unicode_val:04X}) in {font_name}")
        except Exception as e:
            print(f"Error processing font '{font_name}': {e}")
            continue

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(font_vectors, f, ensure_ascii=False, indent=4)

def load_text_from_file(file_path):
    """텍스트 파일에서 텍스트를 읽어옵니다. UTF-8 인코딩을 가정합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            return text
    except FileNotFoundError:
        print(f"Error: Text file not found at {file_path}")
        return "" # 파일이 없을 경우 빈 문자열 반환
    except UnicodeDecodeError:
        print(f"Error: UnicodeDecodeError while reading {file_path}. Please check file encoding.")
        return ""

# 사용 예시
font_folder = "public/fonts"
output_json = "font_vectors.json"
text_file_path = "korean_corpus.txt" # 텍스트 파일 경로

# 텍스트 파일에서 텍스트 로드
loaded_text = load_text_from_file(text_file_path)

if not loaded_text:
    print("Error: Could not load text from file. Using default sample texts.")
    sample_texts = [ # 파일 로드 실패 시 기본 샘플 텍스트 사용
        "안녕하세요. 한국어 폰트 비교입니다.",
        "이 코드는 한글 폰트의 유사성을 분석합니다.",
        "다양한 폰트를 비교하여 원하는 폰트를 찾으세요.",
        "한글 디자인은 매우 아름답습니다.",
        "가나다라마바사아자차카타파하"
    ]
    all_chars = "".join(sample_texts)

else:
    all_chars = loaded_text

# 빈도 기반 샘플링
char_counts = Counter(all_chars)
most_common_chars = [char for char, count in char_counts.most_common(200) if '가' <= char <= '힣']

chars_to_compare = most_common_chars

target_font_path = "public/fonts/DungGeunMo.ttf"

# 벡터화 및 저장 (처음 한 번만 실행)
# vectorize_fonts(font_folder, output_json, chars_to_compare)

if not os.path.exists(target_font_path):
    print(f"Error: Target font file not found: {target_font_path}")
elif not os.path.exists(output_json):
    print(f"Error: JSON file not found: {output_json}")
elif not chars_to_compare: # 비교할 문자가 없을 경우
    print("Error: No characters to compare. Check text file or sample texts.")
else:
    recommend_top_3_fonts(output_json, target_font_path, chars_to_compare)