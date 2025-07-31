import chess
import chess.svg

class ChessGame:
    def __init__(self, fen=None):
        self.initial_fen = fen
        self.board = chess.Board(fen) if fen else chess.Board()
        self.move_history = []
    
    def make_move(self, uci_move):
        move = chess.Move.from_uci(uci_move)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(self.board.san(move))
            return True
        return False
    
    def reset_to_position(self):
        self.board = chess.Board(self.initial_fen) if self.initial_fen else chess.Board()
        self.move_history = []
    
    def get_fen(self):
        return self.board.fen()
    
    def get_move_history(self):
        return " ".join(self.move_history)
    
    def get_board_svg(self, size=400):
        return chess.svg.board(board=self.board, size=size)