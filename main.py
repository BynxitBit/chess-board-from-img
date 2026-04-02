import cv2
import sys
import pygame
from board_detection import process_board_image
from piece_recognition import PieceRecognizer
from game_engine import ChessGame
from gui import ChessGUI
from engine_worker import EngineWorker
import subprocess

def image_to_fen(image_path, active_color="w", debug=False):
    try:
        board_img = process_board_image(image_path)
        cv2.imwrite("cropped_board.jpg", board_img)
        border_size = 1
        board_img = cv2.copyMakeBorder(
            board_img, 
            border_size, border_size, border_size, border_size, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        recognizer = PieceRecognizer("pieces", debug=debug)

        square_size = board_img.shape[0] // 8
        recognizer.calibrate(board_img, square_size, border_size)
        is_flipped = recognizer.detect_orientation(board_img, square_size, border_size)

        # Build a 2D grid of piece codes so we can flip it if needed
        piece_grid = []
        for row in range(8):
            piece_row = []
            for col in range(8):
                y1 = border_size + row * square_size
                y2 = border_size + (row + 1) * square_size
                x1 = border_size + col * square_size
                x2 = border_size + (col + 1) * square_size
                square_img = board_img[y1:y2, x1:x2]
                piece_code = recognizer.recognize_piece(square_img, row, col)
                piece_row.append(piece_code)
            piece_grid.append(piece_row)

        # If board is from Black's perspective (rank 1 at top, h-file on left),
        # flip both axes to produce standard FEN (rank 8 at top, a-file on left).
        if is_flipped:
            piece_grid = [row[::-1] for row in piece_grid][::-1]

        fen_rows = []
        for piece_row in piece_grid:
            fen_row = []
            empty_count = 0
            for piece_code in piece_row:
                if piece_code:
                    if empty_count > 0:
                        fen_row.append(str(empty_count))
                        empty_count = 0
                    fen_piece = piece_code[1].upper() if piece_code[0] == 'w' else piece_code[1].lower()
                    fen_row.append(fen_piece)
                else:
                    empty_count += 1
            if empty_count > 0:
                fen_row.append(str(empty_count))
            fen_rows.append("".join(fen_row))

        fen = "/".join(fen_rows)
        fen += f" {active_color} KQkq - 0 1"
        return fen, is_flipped

    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        raise e

def _copy_to_clipboard(text):
    for cmd in (
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
    ):
        try:
            subprocess.run(cmd, input=text.encode(), check=True, timeout=2)
            return
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    print(f"Clipboard unavailable. Move history: {text}")


def main(image_path=None, active_color="w"):
    fen = f"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR {active_color} KQkq - 0 1"
    is_flipped = False

    if image_path:
        try:
            fen, is_flipped = image_to_fen(image_path, active_color=active_color, debug=True)
            print(f"Detected FEN: {fen}")
            print(f"Board perspective: {'Black' if is_flipped else 'White'}")
        except Exception as e:
            print(f"Error processing image: {e}")
            print("Using default starting position")

    game = ChessGame(fen)
    gui = ChessGUI(game, flipped=is_flipped)

    worker = EngineWorker()
    worker.request_analysis(game.get_fen())

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                action = gui.handle_click(event.pos)

                if action == "RESET":
                    game.reset_to_position()
                    worker.request_analysis(game.get_fen())
                elif action == "MOVE":
                    worker.request_analysis(game.get_fen())
                elif action == "COPY":
                    _copy_to_clipboard(game.get_move_history())

        gui.update_analysis(worker.get_result())
        gui.screen.fill((255, 255, 255))
        gui.draw_board()
        gui.draw_analysis_panel(worker.is_available())
        pygame.display.flip()

    worker.quit()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    active_color = "b" if len(sys.argv) > 2 and sys.argv[2] == "1" else "w"
    main(image_path, active_color)