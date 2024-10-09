import os
import re
import urllib.parse
import pandas as pd
from datetime import datetime

class StructuredDataFromLogFiles:
    def __init__(self, dir_path):
        """
        Initialize the StructuredDataFromLogFiles object with the directory path containing log files.

        Args:
            dir_path (str): Path to the directory containing log files.
        """
        self.dir_path = dir_path

    def sort_log_files(self):
        """
        Sort log files based on their timestamp.

        Returns:
            list: List of sorted log entries.
        """
        log_files = os.listdir(self.dir_path)
        log_entries = []

        def parse_log_file(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.split(' | ')
                    if len(parts) > 1:
                        timestamp = parts[0]
                        log_entries.append((timestamp, line.strip()))

        for log_file in log_files:
            parse_log_file(os.path.join(self.dir_path, log_file))

        log_entries.sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S,%f'))
        return [entry[1] for entry in log_entries]

    def divide_into_chunks(self, log_lines):
        """
        Divide log lines into chunks based on the start of a new request.

        Args:
            log_lines (list): List of log lines.

        Returns:
            list: List of log line chunks.
        """
        chunks = []
        current_chunk = []
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

        for line in log_lines:
            if date_pattern.match(line) and "GET /get?msg=" in line:
                if current_chunk:
                    current_chunk.append(line)
                    chunks.append(current_chunk)
                    current_chunk = []
                else:
                    current_chunk.append(line)
            else:
                current_chunk.append(line)

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def extract_date_time_and_ip_address(self, chunk):
        """
        Extract date, time, and IP address from a log chunk.

        Args:
            chunk (list): List of log lines in a chunk.

        Returns:
            dict: Dictionary with IP address and request date-time.
        """
        log_request_line = chunk[-1]
        pattern = re.compile(r'\|\s(?P<ip_address>\d+\.\d+\.\d+\.\d+)\s-\s-\s\[(?P<request_datetime>[^\]]+)\]')
        match = pattern.search(log_request_line)
        return match.groupdict()

    def extract_question_and_corpus_name(self, chunk):
        """
        Extract question and corpus name from a log chunk.

        Args:
            chunk (list): List of log lines in a chunk.

        Returns:
            tuple: Question and corpus name.
        """
        line = chunk[-1]
        if "GET" in line:
            query_start = line.find("?msg=")
            if query_start != -1:
                query_end = line.find(" HTTP/1.1")
                request_url = line[query_start + 5: query_end]
                try:
                    request = urllib.parse.unquote(request_url)
                    question, corpus_name = request.split('&')
                except ValueError:
                    request = urllib.parse.unquote(request_url)
                    question, corpus_name, _ = request.split('&')
                return question, corpus_name.split('=')[1]
        return None, None

    def check_for_error_response(self, chunk):
        """
        Check for error response in a log chunk.

        Args:
            chunk (list): List of log lines in a chunk.

        Returns:
            str or bool: Error message if found, otherwise False.
        """
        for line in chunk:
            if "Error response from server" in line:
                return "Error response from server"
            elif 'Exception Occurred!' in line:
                return 'Exception Occurred!'
        return False

    def extract_youtube_links(self, chunk):
        """
        Extract the first five YouTube video links from a log chunk.

        Args:
            chunk (list): List of log lines in a chunk.

        Returns:
            list: List of YouTube video links.
        """
        error_message = self.check_for_error_response(chunk)
        if error_message:
            return [error_message]
        
        youtube_links = []
        for line in chunk:
            youtube_match = re.findall(r'(https?://(?:www\.)?youtube\.com/\S+)', line)
            if youtube_match:
                youtube_links.extend(youtube_match)
                if len(youtube_links) >= 5:
                    break
        return youtube_links[:5]

    def extract_answers_and_scores(self, chunk):
        """
        Extract answers and scores from a log chunk.

        Args:
            chunk (list): List of log lines in a chunk.

        Returns:
            tuple: List of answers and scores.
        """
        error_message = self.check_for_error_response(chunk)
        if error_message:
            return [error_message], [error_message]
        
        log_lines = chunk[-14:-2]
        answers = []
        scores = []

        i = 0
        while i < len(log_lines):
            if i % 2 == 0:
                answer = log_lines[i].split('INFO |', 1)[1].strip()
                answers.append(answer)
            else:
                score = log_lines[i].split('INFO |', 1)[1].strip()
                scores.append(score)
            i += 1

        return answers, scores

    def process_logs(self):
        """
        Process all log files and return structured data in a DataFrame.

        Returns:
            DataFrame: DataFrame containing structured data.
        """
        log_lines = self.sort_log_files()
        chunks = self.divide_into_chunks(log_lines)
        result = []

        for chunk in chunks:
            question, corpus = self.extract_question_and_corpus_name(chunk)
            if question:
                data = self.extract_date_time_and_ip_address(chunk)
                ip_address = data.get('ip_address')
                date_time = data.get('request_datetime')
                YouTubeVideos = self.extract_youtube_links(chunk)
                answers, scores = self.extract_answers_and_scores(chunk)
                result.append([question, corpus, ip_address, date_time, YouTubeVideos, answers, scores])

        Data = pd.DataFrame(result)
        Data.columns = ["Question", "Corpus", "Ip Address", "Date and Time of Request", "YouTubeVideos", "Answers", "Scores"]
        Data.reset_index(drop=True, inplace=True)
        return Data


