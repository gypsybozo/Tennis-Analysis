# Tennis Match Analysis Tool

This project provides an in-depth analysis of tennis match footage, focusing on player and ball tracking, court mapping, speed calculations, and visualization of key statistics. Designed to enhance strategic insights, the tool displays a minicourt for simplified viewing of movement patterns and presents real-time speed metrics.

## Features

- **Player and Ball Tracking**: Accurately tracks the positions of players and the tennis ball on the court in real-time.
- **Court Keypoint Mapping**: Maps out essential court points to define boundaries and enhance the precision of player and ball tracking.
- **Minicourt Visualization**: Displays a scaled-down version of the court on the top right of the screen, showing real-time positions and movements of players and the ball.
- **Speed Metrics and Statistics**:
  - **Player Speed**: Real-time display of each player's speed.
  - **Shot Speed**: Records and displays shot speeds after every hit.
  - **Average Shot Speed**: Calculates and updates the average shot speed for each player.
  - **Average Player Speed**: Updates the average movement speed of each player throughout the match.
- **Intuitive User Interface**: The speed metrics and averages for both players are prominently shown on the bottom right, providing quick insights.

## How It Works

1. **Tracking Algorithms**: Using computer vision algorithms, the tool continuously monitors player and ball movements across the court.
2. **Court Mapping**: Key points of the court are mapped to define the boundaries and regions for accurate analysis.
3. **Real-time Minicourt Display**: The minicourt replicates the primary court’s dimensions, simplifying the observation of movement patterns and helping track court coverage.
4. **Speed Calculations**: Speed metrics are calculated based on frame-by-frame analysis, updated live, and displayed for easy reference.
  
## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Pandas
- Ultralytics

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tennis-analysis-tool.git
   cd tennis-analysis-tool
   ```
2. Install the dependencies.
3. Run the Script
   ```bash
   python main.py
   ```
## Future Enhancements
- Support for additional metrics (e.g., rally length, player stamina).
- Interactive visualization options for customized playback.
- Integration with other data sources for enhanced insights.
