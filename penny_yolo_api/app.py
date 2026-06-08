"""
Penny English - YOLOv8 Object Detection API
Deploy on Render.com
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

model = None

def load_model():
    global model
    try:
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
        import torch
        torch.serialization.add_safe_globals([DetectionModel])
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

load_model()

VOCAB_MAP = {
    'person':           {'vi': 'người',                 'phonetic': '/ˈpɜːrsən/'},
    'bicycle':          {'vi': 'xe đạp',                'phonetic': '/ˈbaɪsɪkəl/'},
    'car':              {'vi': 'xe ô tô',               'phonetic': '/kɑːr/'},
    'motorcycle':       {'vi': 'xe máy',                'phonetic': '/ˈmoʊtərsaɪkəl/'},
    'airplane':         {'vi': 'máy bay',               'phonetic': '/ˈerpleɪn/'},
    'bus':              {'vi': 'xe buýt',               'phonetic': '/bʌs/'},
    'train':            {'vi': 'tàu hỏa',               'phonetic': '/treɪn/'},
    'truck':            {'vi': 'xe tải',                'phonetic': '/trʌk/'},
    'boat':             {'vi': 'thuyền',                'phonetic': '/boʊt/'},
    'traffic light':    {'vi': 'đèn giao thông',        'phonetic': '/ˈtræfɪk laɪt/'},
    'fire hydrant':     {'vi': 'cột chữa cháy',         'phonetic': '/faɪər ˈhaɪdrənt/'},
    'stop sign':        {'vi': 'biển báo dừng',         'phonetic': '/stɒp saɪn/'},
    'parking meter':    {'vi': 'đồng hồ đỗ xe',         'phonetic': '/ˈpɑːrkɪŋ ˈmiːtər/'},
    'bench':            {'vi': 'ghế băng',              'phonetic': '/bentʃ/'},
    'bird':             {'vi': 'chim',                  'phonetic': '/bɜːrd/'},
    'cat':              {'vi': 'mèo',                   'phonetic': '/kæt/'},
    'dog':              {'vi': 'chó',                   'phonetic': '/dɒɡ/'},
    'horse':            {'vi': 'ngựa',                  'phonetic': '/hɔːrs/'},
    'sheep':            {'vi': 'cừu',                   'phonetic': '/ʃiːp/'},
    'cow':              {'vi': 'bò',                    'phonetic': '/kaʊ/'},
    'elephant':         {'vi': 'voi',                   'phonetic': '/ˈelɪfənt/'},
    'bear':             {'vi': 'gấu',                   'phonetic': '/ber/'},
    'zebra':            {'vi': 'ngựa vằn',              'phonetic': '/ˈziːbrə/'},
    'giraffe':          {'vi': 'hươu cao cổ',           'phonetic': '/dʒɪˈræf/'},
    'backpack':         {'vi': 'ba lô',                 'phonetic': '/ˈbækpæk/'},
    'umbrella':         {'vi': 'ô / dù',                'phonetic': '/ʌmˈbrelə/'},
    'handbag':          {'vi': 'túi xách',              'phonetic': '/ˈhændbæɡ/'},
    'tie':              {'vi': 'cà vạt',                'phonetic': '/taɪ/'},
    'suitcase':         {'vi': 'vali',                  'phonetic': '/ˈsuːtkeɪs/'},
    'frisbee':          {'vi': 'đĩa ném frisbee',       'phonetic': '/ˈfrɪzbiː/'},
    'skis':             {'vi': 'ván trượt tuyết',       'phonetic': '/skiːz/'},
    'snowboard':        {'vi': 'ván trượt tuyết',       'phonetic': '/ˈsnoʊbɔːrd/'},
    'sports ball':      {'vi': 'bóng thể thao',         'phonetic': '/spɔːrts bɔːl/'},
    'kite':             {'vi': 'diều',                  'phonetic': '/kaɪt/'},
    'baseball bat':     {'vi': 'gậy bóng chày',         'phonetic': '/ˈbeɪsbɔːl bæt/'},
    'baseball glove':   {'vi': 'găng tay bóng chày',   'phonetic': '/ˈbeɪsbɔːl ɡlʌv/'},
    'skateboard':       {'vi': 'ván trượt',             'phonetic': '/ˈskeɪtbɔːrd/'},
    'surfboard':        {'vi': 'ván lướt sóng',         'phonetic': '/ˈsɜːrfbɔːrd/'},
    'tennis racket':    {'vi': 'vợt tennis',            'phonetic': '/ˈtenɪs ˈrækɪt/'},
    'bottle':           {'vi': 'chai',                  'phonetic': '/ˈbɒtəl/'},
    'wine glass':       {'vi': 'ly rượu',               'phonetic': '/waɪn ɡlɑːs/'},
    'cup':              {'vi': 'cốc',                   'phonetic': '/kʌp/'},
    'fork':             {'vi': 'dĩa',                   'phonetic': '/fɔːrk/'},
    'knife':            {'vi': 'dao',                   'phonetic': '/naɪf/'},
    'spoon':            {'vi': 'thìa',                  'phonetic': '/spuːn/'},
    'bowl':             {'vi': 'bát',                   'phonetic': '/boʊl/'},
    'banana':           {'vi': 'chuối',                 'phonetic': '/bəˈnɑːnə/'},
    'apple':            {'vi': 'táo',                   'phonetic': '/ˈæpəl/'},
    'sandwich':         {'vi': 'bánh sandwich',         'phonetic': '/ˈsænwɪtʃ/'},
    'orange':           {'vi': 'cam',                   'phonetic': '/ˈɒrɪndʒ/'},
    'broccoli':         {'vi': 'bông cải xanh',         'phonetic': '/ˈbrɒkəli/'},
    'carrot':           {'vi': 'cà rốt',                'phonetic': '/ˈkærət/'},
    'hot dog':          {'vi': 'xúc xích',              'phonetic': '/hɒt dɒɡ/'},
    'pizza':            {'vi': 'pizza',                 'phonetic': '/ˈpiːtsə/'},
    'donut':            {'vi': 'bánh donut',            'phonetic': '/ˈdoʊnʌt/'},
    'cake':             {'vi': 'bánh kem',              'phonetic': '/keɪk/'},
    'chair':            {'vi': 'ghế',                   'phonetic': '/tʃer/'},
    'couch':            {'vi': 'ghế sofa',              'phonetic': '/kaʊtʃ/'},
    'potted plant':     {'vi': 'cây trồng chậu',        'phonetic': '/ˈpɒtɪd plɑːnt/'},
    'bed':              {'vi': 'giường',                'phonetic': '/bed/'},
    'dining table':     {'vi': 'bàn ăn',                'phonetic': '/ˈdaɪnɪŋ ˈteɪbəl/'},
    'toilet':           {'vi': 'bồn cầu',               'phonetic': '/ˈtɔɪlɪt/'},
    'tv':               {'vi': 'TV / tivi',             'phonetic': '/ˌtiːˈviː/'},
    'laptop':           {'vi': 'máy tính xách tay',     'phonetic': '/ˈlæptɒp/'},
    'mouse':            {'vi': 'chuột máy tính',        'phonetic': '/maʊs/'},
    'remote':           {'vi': 'điều khiển từ xa',      'phonetic': '/rɪˈmoʊt/'},
    'keyboard':         {'vi': 'bàn phím',              'phonetic': '/ˈkiːbɔːrd/'},
    'cell phone':       {'vi': 'điện thoại',            'phonetic': '/sel foʊn/'},
    'microwave':        {'vi': 'lò vi sóng',            'phonetic': '/ˈmaɪkrəweɪv/'},
    'oven':             {'vi': 'lò nướng',              'phonetic': '/ˈʌvən/'},
    'toaster':          {'vi': 'máy nướng bánh mì',     'phonetic': '/ˈtoʊstər/'},
    'sink':             {'vi': 'bồn rửa',               'phonetic': '/sɪŋk/'},
    'refrigerator':     {'vi': 'tủ lạnh',               'phonetic': '/rɪˈfrɪdʒəreɪtər/'},
    'book':             {'vi': 'sách',                  'phonetic': '/bʊk/'},
    'clock':            {'vi': 'đồng hồ',               'phonetic': '/klɒk/'},
    'vase':             {'vi': 'bình hoa',              'phonetic': '/vɑːz/'},
    'scissors':         {'vi': 'kéo',                   'phonetic': '/ˈsɪzərz/'},
    'teddy bear':       {'vi': 'gấu bông',              'phonetic': '/ˈtedi ber/'},
    'hair drier':       {'vi': 'máy sấy tóc',           'phonetic': '/her ˈdraɪər/'},
    'toothbrush':       {'vi': 'bàn chải đánh răng',    'phonetic': '/ˈtuːθbrʌʃ/'},
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'message': 'Penny English YOLOv8 API'
    })

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        results = model(image, conf=0.4)

        detected = []
        seen = set()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]

                if class_name not in seen:
                    seen.add(class_name)
                    vocab = VOCAB_MAP.get(class_name, {
                        'vi': class_name,
                        'phonetic': f'/{class_name}/'
                    })
                    detected.append({
                        'name': class_name,
                        'confidence': round(confidence * 100, 1),
                        'vi': vocab['vi'],
                        'phonetic': vocab['phonetic'],
                    })

        detected.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'success': True,
            'count': len(detected),
            'objects': detected[:10]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
