# AlphaChess

A deep reinforcement learning chess engine implemented in Python and PyTorch, utilizing a self-play pipeline modeled after the AlphaZero architecture. 

The primary objective of this project is to develop an engine that learns chess positioning and move evaluation entirely from ground-up self-play, bypassing traditional human-designed heuristic evaluation functions.

## Architecture

The engine splits computational overhead between raw pattern intuition and lookahead calculation:

* **State Representation:** The current board state is encoded into a 12x8x8 binary tensor. Six channels map the positions of friendly pieces (pawn, knight, bishop, rook, queen, king), and six channels map the opponent's pieces.
* **Dual-Headed Network:** The encoded tensor passes through a series of convolutional layers before splitting into two distinct heads:
  * **Policy Head:** Outputs a 4096-dimensional vector representing raw move probabilities across the board's entry and exit squares.
  * **Value Head:** Outputs a scalar value bounded between -1.0 (Black advantage) and +1.0 (White advantage), estimating the expected game outcome from the current state.

## Core Modules

* `src/encoder.py`: Handles serialization of `python-chess` board states into optimized NumPy arrays and PyTorch tensors.
* `src/model.py`: Defines the dual-headed convolutional network architecture.
* `src/main.py`: Entry point for initializing the network architecture and running pipeline diagnostics.

## Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* NumPy
* python-chess

### Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/ayaanokoji792/AlphaChess-RL.git](https://github.com/ayaanokoji792/AlphaChess-RL.git)
cd AlphaChess-RL
pip install -r requirements.txt
