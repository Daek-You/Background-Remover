# main.py 수정
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)