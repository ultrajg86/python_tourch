# app/controllers/user_controller.py

from flask import Blueprint, jsonify, request

from app.services.user_service import UserService
from app.models.user import User

user_controller = Blueprint('user_controller', __name__)

# 사용자 목록 조회
@user_controller.route('/users', methods=['GET'])
def get_users():
    users = UserService.get_all_users()
    user_list = [user.to_dict() for user in users]
    return jsonify(user_list)

# 사용자 상세 조회
@user_controller.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = UserService.get_user_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(user.to_dict())

# 사용자 생성
@user_controller.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid payload'}), 400
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'error': 'Missing required fields'}), 400

    user = User(username=username, email=email, password=password)
    created_user = UserService.create_user(user)
    return jsonify(created_user.to_dict()), 201

# 사용자 수정
@user_controller.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    user = UserService.get_user_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid payload'}), 400

    username = data.get('username')
    email = data.get('email')

    if not username or not email:
        return jsonify({'error': 'Missing required fields'}), 400

    user.username = username
    user.email = email

    updated_user = UserService.update_user(user)
    return jsonify(updated_user.to_dict())

# 사용자 삭제
@user_controller.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user = UserService.get_user_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    UserService.delete_user(user)
    return jsonify({'message': 'User deleted'})
