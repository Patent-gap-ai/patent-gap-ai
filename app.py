import os
import threading
from flask import Flask, request, jsonify, session
from flask_cors import CORS

from models.cases import *
from models.users import *
from env_controller import *
from controller import *

app = Flask(__name__)
CORS(app, origins="*")
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')


def get_user_id():
    """
    Get user_id from X-User-ID header or session.
    Returns None if not found.
    """
    # First check header
    user_id = request.headers.get('X-User-ID')
    if user_id:
        return user_id
    
    # Fallback to session for backward compatibility
    if 'user_id' in session:
        return session['user_id']
    
    return None


@app.route('/api/fetch-patent-from-uspto', methods=['POST'])
def fetch_patent_from_uspto():
  """
  Fetch a patent from the US Patent Office and create a case
  ---
  tags:
    - Patents
  summary: Fetch patent from USPTO
  description: |
    Fetches patent data from the US Patent and Trademark Office (USPTO) using the provided patent ID.
    The patent data is then normalized and automatically created as a case in the system.
    The case will be monitored for similarity analysis.
  consumes:
    - application/json
  produces:
    - application/json
  security:
    - session: []
  parameters:
    - in: body
      name: patent_request
      description: Patent ID to fetch from USPTO
      required: true
      schema:
        type: object
        required:
          - patentId
        properties:
          patentId:
            type: string
            description: The USPTO patent number/ID to fetch
            example: "US12345678"
  responses:
    200:
      description: Patent fetched successfully and case created
      schema:
        type: object
        properties:
          success:
            type: boolean
            example: true
          message:
            type: string
            example: "Patent has been fetched successfully. This case is now being monitored for similarity."
          case_id:
            type: string
            description: The ID of the created case
            example: "case_12345"
    400:
      description: Bad request - missing or invalid patent ID, or failed to fetch/normalize patent data
      schema:
        type: object
        properties:
          success:
            type: boolean
            example: false
          message:
            type: string
            example: "Patent ID is required"
    500:
      description: Internal server error during patent data processing
      schema:
        type: object
        properties:
          success:
            type: boolean
            example: false
          message:
            type: string
            example: "Error normalizing patent data: [error details]"
  """
  user_id = get_user_id()
  if not user_id:
    print('\nUser ID is not in session')
    return jsonify({'success': False, 'message': 'User ID is not in session'}), 400
  print(f'LOG: {user_id} Import Patent from USPTO')
  
  data = request.get_json()
  if data is None:
    print('\nNo Data provided')
    return jsonify({'success': False, 'message': 'No data provided'}), 400
  if 'patentId' not in data.keys():
    print('\nPatent ID is not Provided')
    return jsonify({'success': False, 'message': 'Patent ID is not provided'}), 400
  patent_id = data.get('patentId')
  if patent_id is None or patent_id == '':
    print('\nPatent ID is not valid')
    return jsonify({'success': False, 'message': 'Patent ID is not valid'}), 400

  # Check if the patent already exists
  if caseAlreadyExists(patent_id, user_id):
    return jsonify({'success': False, 'message': 'Patent already exists in your portfolio'}), 400
  
  user_data = get_user_profile(user_id, show_password=False) 
  fetching_patents = user_data.get('fetching_patents', [])
  patent_id = str(patent_id).strip().upper()
  if patent_id in fetching_patents:
    return jsonify({
        'success': False,
        'message': f'Patent ID {patent_id} is already being fetched by this user.',
        'case_id': patent_id
    }), 409

  print(f'\nFetching patent of ID {patent_id} from USPTO (key: {getEnvKey("uspto")}): {json.dumps(data, indent=4)}')

  update_user_fetching_patents(
    user_id=user_id, 
    fetching_patents=[patent_id], 
    error_patents=[], 
    replace=True
    )
  thread = threading.Thread(
    target=fetchById,
    name=f'{patent_id}-{user_id}',
    args=(app, patent_id, user_id),
    daemon=True,
  )
  thread.start()
  responseBody = {
    'success': True, 
    'message': 'Patent data import started', 
    'case_id': patent_id
  }
  return jsonify(responseBody), 200


@app.route('/api/check-fetching-thread-status/<patent_id>', methods=['GET'])
def check_fetching_thread_status(patent_id):
  user_id = get_user_id()
  if not user_id:
    print('\nUser ID is not in session')
    return jsonify({'success': False, 'message': 'User ID is not in session'}), 400
  print(f'LOG: {user_id} Checking Fetching Thread Status for Patent: {patent_id}')

  status, message = check_thread_status(f'{patent_id}-{user_id}')
  status_code = 200 if status else 400
  return jsonify({'success': status, 'message': message}), status_code


if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', 5000)))