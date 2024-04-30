from flask import Flask

app = Flask(__name__)
# app.register_blueprint(routes_bp)

# if __name__ == '__main__':
#     app.run(debug=True)

# from routes import bp
from app import routes