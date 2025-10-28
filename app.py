from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import uuid
import time, os
from feature_engineer import FeatureEngineer 
app = Flask(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['ua_lower'] = df['user_agent'].str.lower()
        df['ua_length'] = df['user_agent'].str.len()
        df['has_mozilla'] = df['ua_lower'].str.contains('mozilla', na=False)
        df['path_depth'] = df['path'].str.count('/')
        df['is_api_path'] = df['path'].str.contains('storage|api', na=False)
        df['is_static'] = df['path'].str.contains('js|css|ico', na=False)
        return df

# Load the models and encoders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def model_path(filename):
    return os.path.join(BASE_DIR, filename)

# Load the models and encoders (use relative paths)
try:
    loaded_pipeline = joblib.load(model_path('bot_detection_pipeline.pkl'))
    loaded_le = joblib.load(model_path('label_encoder.pkl'))
    print("✅ Primary model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading primary model: {e}")
    loaded_pipeline = None
    loaded_le = None

try:
    mouse_model = joblib.load(model_path("mouse_behavior_rf_model.joblib"))
    mouse_scaler = joblib.load(model_path("mouse_behavior_scaler.joblib"))
    print("✅ Mouse behavior model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading mouse behavior model: {e}")
    mouse_model = None
    mouse_scaler = None

# Store mouse behavior data temporarily (in production, use Redis or database)
user_sessions = {}

def detect_user_type(request_obj):
    """Detect if the current request is from a human or bot"""
    if loaded_pipeline is None or loaded_le is None:
        return 'human', 0  # Fallback to human if model not loaded
    
    try:
        # Extract features from the current request
        method = request_obj.method
        user_agent = request_obj.headers.get('User-Agent', '')
        path = request_obj.path
        referrer = request_obj.headers.get('Referer', '-')
        
        # Estimate bytes (you can make this more sophisticated)
        bytes_val = 512  # Default value
        
        # Check for headless browser indicators
        is_headless = 0
        user_agent_lower = user_agent.lower()
        headless_indicators = ['headless', 'phantom', 'puppeteer', 'selenium', 'playwright']
        if any(indicator in user_agent_lower for indicator in headless_indicators):
            is_headless = 1
        
        # Create DataFrame for prediction
        request_data = pd.DataFrame({
            'method': [method],
            'bytes': [bytes_val],
            'is_headless': [is_headless],
            'user_agent': [user_agent],
            'path': [path],
        })
        
        # Make prediction
        y_pred = loaded_pipeline.predict(request_data)
        y_label = loaded_le.inverse_transform(y_pred)[0]
        
        # Get confidence score
        probabilities = loaded_pipeline.predict_proba(request_data)[0]
        confidence = max(probabilities) * 100
        
        print(f"🔍 Primary Detection Result: {y_label} (Confidence: {confidence:.2f}%)")
        print(f"📊 User Agent: {user_agent}")
        print(f"📍 Path: {path}")
        print(f"🔧 Method: {method}")
        print(f"👻 Headless: {is_headless}")
        
        return y_label, confidence
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return 'human', 0  # Fallback to human on error

def analyze_mouse_behavior(mouse_data):
    """Second layer detection using mouse behavior analysis"""
    if mouse_model is None or mouse_scaler is None:
        return 'human', 0  # Fallback if mouse model not loaded
    
    try:
        # Extract the 3 key features for mouse behavior detection
        values = [
            mouse_data.get('min_step_dist', 0),
            mouse_data.get('angle_variance', 0),
            mouse_data.get('mean_step_dist', 0)
        ]
        
        print(f"🖱️ Mouse behavior features: {values}")
        
        X_scaled = mouse_scaler.transform([values])
        probabilities = mouse_model.predict_proba(X_scaled)[0]
        
        # Simple approach: assume the model returns [human_prob, bot_prob] or vice versa
        # You might need to adjust this based on your actual model's class order
        if len(probabilities) == 2:
            # Assume second class is bot (common convention)
            bot_prob = probabilities[1]
            human_prob = probabilities[0]
            
            # If the model was trained with different class order, check which makes sense
            if bot_prob > human_prob:
                label = 'bot'
                confidence = bot_prob * 100
            else:
                label = 'human' 
                confidence = human_prob * 100
        else:
            # Fallback for unexpected number of classes
            predicted_class = mouse_model.predict(X_scaled)[0]
            label = 'bot' if predicted_class == 1 else 'human'
            confidence = max(probabilities) * 100
        
        print(f"🖱️ Mouse Behavior Detection: {label} (Confidence: {confidence:.2f}%)")
        
        return label, confidence
        
    except Exception as e:
        print(f"❌ Mouse behavior analysis error: {e}")
        return 'human', 0

@app.route('/')
def index():
    """Main entry point - shows mouse behavior analysis page"""
    # Generate session ID for tracking mouse behavior
    session_id = str(uuid.uuid4())[:8]
    user_sessions[session_id] = {
        'created_at': time.time(),
        'primary_detection': None,
        'mouse_analysis': None,
        'final_result': None
    }
    
    return render_template('mouse_analysis.html', session_id=session_id)

@app.route('/analyze-behavior', methods=['POST'])
def analyze_behavior():
    """Endpoint to analyze mouse behavior and make final decision"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        mouse_features = data.get('features', {})
        
        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Perform primary detection first
        primary_type, primary_confidence = detect_user_type(request)
        
        # Store primary detection result
        user_sessions[session_id]['primary_detection'] = {
            'type': primary_type,
            'confidence': primary_confidence
        }
        
        # Perform mouse behavior analysis (second layer)
        mouse_type, mouse_confidence = analyze_mouse_behavior(mouse_features)
        
        # Store mouse analysis result
        user_sessions[session_id]['mouse_analysis'] = {
            'type': mouse_type,
            'confidence': mouse_confidence,
            'features': mouse_features
        }
        
        # Make final decision (you can customize this logic)
        final_type, final_confidence = make_final_decision(
            primary_type, primary_confidence, 
            mouse_type, mouse_confidence
        )
        
        user_sessions[session_id]['final_result'] = {
            'type': final_type,
            'confidence': final_confidence
        }
        
        print(f"🎯 Final Decision: {final_type} (Confidence: {final_confidence:.2f}%)")
        print(f"   - Primary: {primary_type} ({primary_confidence:.2f}%)")
        print(f"   - Mouse: {mouse_type} ({mouse_confidence:.2f}%)")
        
        return jsonify({
            'final_type': final_type,
            'final_confidence': round(final_confidence, 2),
            'primary_detection': {
                'type': primary_type,
                'confidence': round(primary_confidence, 2)
            },
            'mouse_analysis': {
                'type': mouse_type,
                'confidence': round(mouse_confidence, 2)
            },
            'redirect_url': url_for('human_page' if final_type == 'human' else 'bot_page',
                                  user_type=final_type,
                                  confidence=final_confidence,
                                  session_id=session_id)
        })
        
    except Exception as e:
        print(f"❌ Behavior analysis error: {e}")
        return jsonify({'error': str(e)}), 500

def make_final_decision(primary_type, primary_confidence, mouse_type, mouse_confidence):
    """Simplified rule-based final decision"""
    
    print(f"🤖 Decision Making: Primary={primary_type}({primary_confidence}%), Mouse={mouse_type}({mouse_confidence}%)")
    
    # RULE 1: If mouse behavior strongly indicates human, trust it
    if mouse_type == 'human' and mouse_confidence > 60:
        print(f"🎯 Strong human mouse behavior overrides primary detection")
        return 'human', mouse_confidence
    
    # RULE 2: If both agree, use the higher confidence
    if primary_type == mouse_type:
        final_confidence = max(primary_confidence, mouse_confidence)
        print(f"✅ Both methods agree: {primary_type}")
        return primary_type, final_confidence
    
    # RULE 3: If primary detection is bot but mouse shows human, trust mouse more
    if primary_type == 'bot' and mouse_type == 'human':
        # Mouse behavior is harder to fake, so give it more weight
        if mouse_confidence >= 50:
            print(f"🖱️ Human mouse behavior overrides bot primary detection")
            return 'human', mouse_confidence
        else:
            print(f"⚠️ Low-confidence human mouse, defaulting to bot")
            return 'bot', primary_confidence
    
    # RULE 4: If primary detection is human but mouse shows bot, investigate further
    if primary_type == 'human' and mouse_type == 'bot':
        # This could be a sophisticated bot, so be cautious
        if mouse_confidence >= 70:
            print(f"⚠️ High-confidence bot mouse behavior overrides human primary")
            return 'bot', mouse_confidence
        else:
            print(f"✅ Trusting human primary over low-confidence bot mouse")
            return 'human', primary_confidence
    
    # Fallback: Use confidence-weighted average
    primary_weight = primary_confidence / 100
    mouse_weight = mouse_confidence / 100
    total_weight = primary_weight + mouse_weight
    
    bot_score = (primary_weight if primary_type == 'bot' else 0) + (mouse_weight if mouse_type == 'bot' else 0)
    bot_score /= total_weight
    
    final_type = 'bot' if bot_score >= 0.5 else 'human'
    final_confidence = bot_score * 100 if final_type == 'bot' else (1 - bot_score) * 100
    
    print(f"⚖️ Fallback weighted decision: {final_type} ({final_confidence:.1f}%)")
    return final_type, final_confidence

@app.route('/human')
def human_page():
    """Page shown when human is detected"""
    user_type = request.args.get('user_type', 'human')
    confidence = request.args.get('confidence', '0')
    session_id = request.args.get('session_id', 'unknown')
    
    # Get session data if available
    session_data = user_sessions.get(session_id, {})
    
    # Get current request info for display
    current_request = {
        'user_agent': request.headers.get('User-Agent', 'Unknown'),
        'ip': request.remote_addr
    }
    
    return render_template('human.html', 
                         user_type=user_type,
                         confidence=confidence,
                         session_id=session_id,
                         session_data=session_data,
                         current_request=current_request)

@app.route('/bot')
def bot_page():
    """Page shown when bot is detected"""
    user_type = request.args.get('user_type', 'bot')
    confidence = request.args.get('confidence', '0')
    session_id = request.args.get('session_id', 'unknown')
    
    # Get session data if available
    session_data = user_sessions.get(session_id, {})
    
    # Get current request info for display
    current_request = {
        'user_agent': request.headers.get('User-Agent', 'Unknown'),
        'ip': request.remote_addr
    }
    
    return render_template('bot.html', 
                         user_type=user_type,
                         confidence=confidence,
                         session_id=session_id,
                         session_data=session_data,
                         current_request=current_request)

@app.route('/admin')
def admin_page():
    """Admin page to see current request details (for testing)"""
    user_type, confidence = detect_user_type(request)
    
    request_info = {
        'ip': request.remote_addr,
        'user_agent': request.headers.get('User-Agent'),
        'method': request.method,
        'path': request.path,
        'referrer': request.headers.get('Referer', 'Direct access'),
        'headers': {key: value for key, value in request.headers}
    }
    
    return render_template('admin.html',
                         request_info=request_info,
                         user_type=user_type,
                         confidence=confidence)

# Keep all your existing API endpoints (they remain the same)
@app.route('/api/detect', methods=['GET', 'POST'])
def api_detect():
    """API endpoint for manual testing"""
    user_type, confidence = detect_user_type(request)
    
    return jsonify({
        'user_type': user_type,
        'confidence': round(confidence, 2),
        'request_info': {
            'ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent'),
            'method': request.method,
            'path': request.path
        }
    })

@app.route('/test-bot')
def test_bot():
    """Test endpoint that simulates bot behavior"""
    try:
        # Create bot-like request data
        bot_data = create_bot_request_data()
        
        # Create a mock request object for prediction
        class MockRequest:
            def __init__(self, data):
                self.method = data['method']
                self.path = data['path']
                self.remote_addr = '127.0.0.5'
                self._headers = {'User-Agent': data['user_agent']}
            
            def headers_get(self, key, default=None):
                return self._headers.get(key, default)
            
            # Make it compatible with our detection function
            @property
            def headers(self):
                class Headers:
                    def get(self, key, default=None):
                        return self._headers.get(key, default)
                headers_obj = Headers()
                headers_obj._headers = self._headers
                return headers_obj
        
        mock_request = MockRequest(bot_data)
        
        # Manually create prediction data since we can't use the mock directly
        prediction_data = pd.DataFrame({
            'method': [bot_data['method']],
            'bytes': [bot_data['bytes']],
            'is_headless': [bot_data['is_headless']],
            'user_agent': [bot_data['user_agent']],
            'path': [bot_data['path']],
        })
        
        # Make prediction
        y_pred = loaded_pipeline.predict(prediction_data)
        y_label = loaded_le.inverse_transform(y_pred)[0]
        probabilities = loaded_pipeline.predict_proba(prediction_data)[0]
        confidence = max(probabilities) * 100
        
        return jsonify({
            'simulation': 'bot',
            'detected_as': y_label,
            'confidence': round(confidence, 2),
            'simulated_data': bot_data,
            'message': 'Bot simulation completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Bot simulation failed'
        }), 500

@app.route('/test-human')
def test_human():
    """Test endpoint that uses current request (should be detected as human)"""
    user_type, confidence = detect_user_type(request)
    
    return jsonify({
        'simulation': 'human',
        'detected_as': user_type,
        'confidence': round(confidence, 2),
        'user_agent': request.headers.get('User-Agent'),
        'message': 'Human detection test completed'
    })

@app.route('/manual-test', methods=['POST'])
def manual_test():
    """Endpoint for manual testing with custom parameters"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Create prediction data from manual input
        prediction_data = pd.DataFrame({
            'method': [data.get('method', 'GET')],
            'bytes': [data.get('bytes', 512)],
            'is_headless': [data.get('is_headless', 0)],
            'user_agent': [data.get('user_agent', '')],
            'path': [data.get('path', '/')],
        })
        
        # Make prediction
        y_pred = loaded_pipeline.predict(prediction_data)
        y_label = loaded_le.inverse_transform(y_pred)[0]
        probabilities = loaded_pipeline.predict_proba(prediction_data)[0]
        confidence = max(probabilities) * 100
        
        return jsonify({
            'prediction': y_label,
            'confidence': round(confidence, 2),
            'probabilities': {
                'bot': round(probabilities[0] * 100, 2),
                'human': round(probabilities[1] * 100, 2)
            },
            'input_data': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_bot_request_data():
    """Create request data that simulates bot behavior"""
    bot_user_agents = [
        'Python-urllib/3.8',
        'curl/7.68.0', 
        'Go-http-client/1.1',
        'Java/1.8.0_291',
        'node-fetch/2.6.1',
        'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
        'Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)'
    ]
    
    import random
    bot_ua = random.choice(bot_user_agents)
    
    return {
        'method': 'GET',
        'bytes': 449,
        'is_headless': 1,
        'user_agent': bot_ua,
        'path': '/api/data'
    }

if __name__ == '__main__':
    # Local debug/dev only
    debug_mode = os.environ.get('FLASK_DEBUG', 'False') == 'True'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
