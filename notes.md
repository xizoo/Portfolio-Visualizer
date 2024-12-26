# Python Financial Analytics Project
- **Author**: Toh Xian Zong
-  **Last Updated**: 17/12/2024
- **Project Overview**: A Finance Toolkit that is able to extract, visualise data of stocks as well as develop models to analyse the data.
- **Credits**: Thanks to [yfinance](https://pypi.org/project/yfinance/) for simplifying the API process

## Key Achievements

### 1. Data Extractor

- **Date Completed**: 17/12/2024
- **Notes**:
    - The package is now able to extract:
        - Balance sheet
        - Price history
        - Return percentages (interval returns)
    - The program is user-friendly:
        - Users can specify custom durations (e.g., 1y, 10y, max) and intervals (e.g., 1mo, 1wk).
        - Simple and clean interface via `run.py`.
    - Outputs are saved in a clean `data/` folder with standardized, timestamped filenames.
    - The terminal prints clear and easy-to-understand instructions.

- **Learning Points**:
    - **User-Friendliness**: Keep the user in mind. A program must be intuitive and easy to run.
    - **Customizability**: Users want flexibility. Prompt for durations, interval, and price types to meet specific needs.
    - **Building Blocks and Wishful Thinking**: To end up with a final program, we need to start small and build up. The development process needs to have a starting point, sall interval steps and a endgoal.

- **Development Process**:
    1. Started with a simple **returns extractor** as a function.
    2. Turned it into a **class object** (`StockData`) to add more features like *price* and *balance sheets*.
    3. Added more **duration support** for up to `max` (IPO to now) for long-term analysis, for users who want large datasets.
    4. Made it **user-friendly** by allowing users to type "options" to display all available durations.
    5. Split the program into **frontend (run.py)** and **backend (stockinfo.py)**.
    6. Cleaned up file naming and organized outputs into folders for clarity.


#### 2. Problems Faced and Solutions
| **Problem**                               | **Root Cause**                       | **Solution**                                         |
|-------------------------------------------|--------------------------------------|-----------------------------------------------------|
| **OSError: Non-Existent Directory**       | Incorrect relative path handling     | Used `os.path.abspath()` for absolute path creation.|
| **Unnecessary Conversion of IPO → max**   | Added extra logic for "ipo" duration | Simplified by asking users to input `max` directly. |
| **File Naming Issues for Balance Sheet**  | Placeholder "N/A" in file names      | Cleaned file names by passing empty strings.        |
| **User Confusion on Durations**           | Lack of clear duration guidance      | Added an "options" command to display valid inputs. |
| **Empty Balance Sheet Data**              | Incomplete data from `yfinance`      | Added checks to ensure non-empty data before saving.|


#### 3. Notes on Final Implementation
- **Features**:
    - Extract balance sheet, price history, and return percentages.
    - Allow users to specify:
        - Custom durations: 1d, 1mo, 1y, 5y, 10y, 30y, max.
        - Intervals: 1d, 1wk, 1mo.
    - Added a user-friendly "options" command to display valid durations.

- **Code Organization**:
    - **`run.py`**: User-facing script that handles all inputs and outputs.
    - **`resources/stockinfo.py`**: Backend logic for fetching, processing, and saving stock data.
    - **`data/`**: All CSV outputs are saved here.

- **Output Files**:
    - Standardized filenames with timestamps:
      ```
      <ticker>_<datatype>_<timestamp>_<duration>_<interval>.csv
      ```

#### 4. Learning Points
- **User-Friendliness**:
    - A good program is easy to use and intuitive.
    - Clear prompts, instructions, and options make a big difference.

- **Customizability**:
    - Users want flexibility. Always prompt for durations, intervals, and price types.

- **Incremental Development**:
    - Build step-by-step:
        - Start small with an MVP (minimum viable product).
        - Add features incrementally (e.g., returns → price → balance sheet).

- **Error Handling**:
    - Common errors like path issues and empty data can arise when using unfamiliar libraries.
    - Thorough testing and learning to read documentation are crucial.

- **Balancing Simplicity and Efficiency**:
    - Avoid over-engineering. Simplifying logic (e.g., direct "max" input) makes the program cleaner and faster.