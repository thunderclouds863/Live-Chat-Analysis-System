# Setup & Configuration
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import json
import os
import joblib
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
warnings.filterwarnings('ignore')

# Configuration
class config:
    # File paths
    RAW_DATA_PATH = "data/raw_conversation.xlsx"
    COMPLAINT_DATA_PATH = "data/complaint_data.xlsx"
    PROCESSED_DATA_PATH = "data/processed/conversations.pkl"
    MODEL_SAVE_PATH = "models/"
    
    # ML Configuration
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Time thresholds (dalam menit)
    NORMAL_THRESHOLD = 5
    SERIOUS_FIRST_REPLY_THRESHOLD = 5
    SERIOUS_FINAL_REPLY_THRESHOLD = 480  # 8 jam
    COMPLAINT_FINAL_REPLY_THRESHOLD = 7200  # 5 hari
    
    # Abandoned detection
    ABANDONED_TIMEOUT_MINUTES = 30  # 30 menit tanpa response dari customer
    CUSTOMER_LEAVE_TIMEOUT = 3  # 3 menit untuk detect customer leave - DIPERBAIKI
    
    # Keywords
    TICKET_REOPENED_KEYWORD = "Ticket Has Been Reopened by"
    CUSTOMER_LEAVE_KEYWORD = "Mohon maaf, dikarenakan tidak ada respon, chat ini Kami akhiri. Terima kasih telah menggunakan layanan Live Chat Toyota Astra Motor, selamat beraktivitas kembali."
    OPERATOR_GREETING_KEYWORDS = [
        "Selamat pagi", "Selamat siang", "Selamat sore", "Selamat malam",
        "Selamat datang di layanan Live Chat Toyota Astra Motor"
    ]
    
    # Action keywords untuk serious first reply - DIPERBANYAK
    ACTION_KEYWORDS = [
        "bantu koordinasikan", "bantu", "koordinasikan", "diteruskan", "disampaikan", "dihubungi", "dicek", "dipelajari",
        "ditindaklanjuti", "dilakukan pengecekan", "dibantu", "dikonsultasikan",
        "dikoordinasikan", "dilaporkan", "dievaluasi", "dianalisis", "diperbaiki",
        "diatasi", "diselesaikan", "ditangani", "diperhatikan", "direspons",
        "diberikan solusi", "diberikan penjelasan", "diberikan informasi",
        "dilakukan pemeriksaan", "dilakukan pengecekan ulang", "dilakukan perbaikan",
        "akan diteruskan", "akan disampaikan", "akan dihubungi", "akan dicek",
        "sedang diproses", "sedang ditindaklanjuti", "dalam penanganan",
        "tim terkait", "pihak terkait", "akan dikonsultasikan", "akan dilaporkan",
        "proses pengecekan", "proses penanganan", "proses penyelesaian"
    ]
     
    REASSIGNED_KEYWORD = "reassigned to"
    
    # Solution keywords untuk normal reply
    SOLUTION_KEYWORDS = [
        'solusi', 'jawaban', 'caranya', 'prosedur', 'bisa menghubungi', 'silakan menghubungi', 
        'disarankan untuk', 'rekomendasi', 'bisa dilakukan', 'langkah', 'cara', 'solusinya',
        'penyelesaian', 'penanganan', 'informasi', 'dijelaskan', 'diberitahu', 'diberikan',
        'dapat dilakukan', 'bisa dengan', 'dapat diatasi', 'cara mengatasi', 'penyebab',
        'perbaikan', 'tindakan', 'rekomendasi', 'saran', 'anjuran', 'instruksi'
    ]
    
    # Closing message patterns untuk dihindari sebagai final reply
    CLOSING_PATTERNS = [
        'informasi cukup', 'informasi cukup?', 'dibantu kembali', 'bantu kembali', 'ada hal lain', 'apabila sudah cukup', 'apakah sudah cukup', 'apakah informasinya sudah cukup',
        'terima kasih telah menghubungi', 'selamat beraktivitas', 'goodbye', 'bye',
        'sampai jumpa', 'terima kasih', 'semoga membantu', 'jika ada pertanyaan',
        'jika masih ada pertanyaan', 'demikian informasi', 'baik [a-zA-Z]+,',
        'terimakasih', 'terima kasih banyak', 'selamat [a-zA-Z]+', 'thank you',
        'thanks', 'good night', 'good day', 'have a nice day', 'sampai jumpa kembali',
        'ditunggu kabarnya', 'ditunggu konfirmasinya', 'mohon ditunggu'
    ]

    VAGUE_QUESTION_PATTERNS = [
        'mau nanya', 'mau tanya', 'boleh tanya', 'saya ingin bertanya', 'saya mau tanya',
        'halo', 'hai', 'selamat', 'pagi', 'siang', 'sore', 'malam', 'permisi'
    ]

    IRRELEVANT_MAIN_QUESTION_PATTERNS = [
        # Promotional messages
        r'kode undian', r'hadiah', r'berkesempatan', r'pemenang', r'lucky draw', r'toyota space',
        r'gr 2025', r'innova zenix', r'yaris cross', r'asuransi gratis', r'klaim hadiah',
        r'pengundian', r'gandaria city', r'instagram @toyotaid', r'spk periode',
        
        # Greetings and system messages
        r'^halo\s+[A-Z]+,?$', r'^selamat\s+(pagi|siang|sore|malam)', r'^dengan\s+\w+',
        r'^ada\s+yang\s+bisa\s+dibantu', r'^boleh\s+dibantu', r'^bisa\s+dibantu',
        r'^saya\s+mau\s+tanya$', r'^mau\s+tanya$', r'^boleh\s+tanya$',
        r'^permisi', r'^halo$', r'^hai$',
        
        # Very short/vague questions
        r'^apa\s+bisa\s+dibantu$', r'^bisa\s+dibantu$', r'^untuk\s+rush\s+gr\s+2025$'
    ]

    MEANINGFUL_QUESTION_MIN_WORDS = 5
    MEANINGFUL_QUESTION_MIN_CHARS = 20

# Data Preprocessor
class DataPreprocessor:
    def __init__(self):
        self.role_mapping = {
            'bot': 'Bot',
            'customer': 'Customer', 
            'operator': 'Operator',
            'ticket automation': 'Ticket Automation',
            '': 'Blank'
        }
    
    def load_raw_data(self, file_path):
        """Load data dari Excel dengan format yang ditentukan"""
        print(f"📖 Loading data from {file_path}")
        
        try:
            df = pd.read_excel(file_path)
            print(f"✅ Loaded {len(df)} rows")
            
            # Validasi columns
            required_columns = ['No', 'Ticket Number', 'Role', 'Sender', 'Message Date', 'Message']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"⚠️ Missing columns: {missing_columns}")
                return None
            
            # Clean data
            df = self.clean_data(df)
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def load_complaint_data(self, file_path):
        """Load complaint data untuk matching"""
        print(f"📖 Loading complaint data from {file_path}")
        
        try:
            df = pd.read_excel(file_path)
            print(f"✅ Loaded {len(df)} complaint records")
            
            # Clean phone numbers
            if 'No.Handphone' in df.columns:
                df['Cleaned_Phone'] = df['No.Handphone'].astype(str).str.replace(r'\D', '', regex=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading complaint data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean dan preprocess data - DENGAN VALIDATION"""
        if df is None or df.empty:
            print("❌ Input data is None or empty")
            return None
        
        try:
            # Copy dataframe
            df_clean = df.copy()
            
            print(f"🧹 Starting cleaning process: {len(df_clean)} rows")
            
            # Handle missing values - �VALIDATION
            initial_count = len(df_clean)
            df_clean = df_clean.dropna(subset=['Message', 'Ticket Number'])
            after_missing = len(df_clean)
            print(f"📝 After dropping missing values: {after_missing}/{initial_count} rows")
            
            if df_clean.empty:
                print("❌ No data left after dropping missing values")
                return None
            
            # Clean text
            df_clean['Message'] = df_clean['Message'].astype(str).str.strip()
            
            # Parse timestamp
            df_clean['parsed_timestamp'] = pd.to_datetime(
                df_clean['Message Date'], errors='coerce'
            )
            
            # Remove invalid timestamps
            initial_timestamp = len(df_clean)
            df_clean = df_clean[df_clean['parsed_timestamp'].notna()]
            final_timestamp = len(df_clean)
            print(f"📅 Valid timestamps: {final_timestamp}/{initial_timestamp}")
            
            if df_clean.empty:
                print("❌ No data left after timestamp validation")
                return None
            
            # Standardize roles
            df_clean['Role'] = df_clean['Role'].str.lower().map(
                lambda x: self.role_mapping.get(x, x.title())
            )
            
            # Fill blank roles dengan 'Blank'
            df_clean['Role'] = df_clean['Role'].fillna('Blank')
            
            # Filter meaningful messages
            df_clean = df_clean[df_clean['Message'].str.len() > 1]
            
            print(f"✅ Final cleaned data: {len(df_clean)} rows")
            
            # 🆕 FINAL VALIDATION
            if df_clean.empty:
                print("❌ No valid data after all cleaning steps")
                return None
            
            return df_clean
            
        except Exception as e:
            print(f"❌ Error during data cleaning: {e}")
            return None
        
    def extract_customer_info(self, df):
        """Extract customer phone dan name dari setiap ticket"""
        customer_info = {}
        
        for ticket_id in df['Ticket Number'].unique():
            ticket_df = df[df['Ticket Number'] == ticket_id]
            
            # Cari di row pertama ticket untuk phone/name
            first_row = ticket_df.iloc[0]
            
            # Check semua kolom untuk phone/name patterns
            phone = None
            name = None
            
            for col in ticket_df.columns:
                if pd.api.types.is_string_dtype(ticket_df[col]):
                    # Cari phone patterns
                    phone_patterns = [r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b', r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b']
                    for pattern in phone_patterns:
                        matches = ticket_df[col].astype(str).str.extract(f'({pattern})', expand=False)
                        if not matches.isna().all():
                            phone = matches.dropna().iloc[0] if not matches.dropna().empty else None
                            break
                    
                    # Cari name patterns (asumsi di kolom Sender atau kolom teks lainnya)
                    if 'sender' in col.lower() or 'name' in col.lower():
                        name_candidates = ticket_df[col].astype(str).str.extract(r'([A-Za-z\s]{3,})', expand=False)
                        if not name_candidates.isna().all():
                            name = name_candidates.dropna().iloc[0] if not name_candidates.dropna().empty else None
            
            customer_info[ticket_id] = {
                'phone': phone,
                'name': name,
                'source': first_row.get('Source', 'Unknown') if 'Source' in ticket_df.columns else 'Unknown'
            }
        
        return customer_info

    def match_complaint_tickets(self, raw_df, complaint_df):
        """Match tickets antara raw data dan complaint data - DENGAN VALIDATION"""
        complaint_tickets = {}
        
        # 🆕 VALIDATION: Check input data
        if raw_df is None or raw_df.empty:
            print("❌ Raw data is None or empty for complaint matching")
            return complaint_tickets
            
        if complaint_df is None or complaint_df.empty:
            print("⚠️ No complaint data provided")
            return complaint_tickets
        
        if 'Cleaned_Phone' not in complaint_df.columns:
            print("⚠️ No Cleaned_Phone column in complaint data")
            return complaint_tickets
        
        try:
            # Extract phones dari raw data
            raw_phones = self._extract_phones_from_raw_data(raw_df)
            
            for _, complaint_row in complaint_df.iterrows():
                complaint_phone = complaint_row['Cleaned_Phone']
                
                if pd.isna(complaint_phone) or complaint_phone == 'nan':
                    continue
                    
                # Cari matching phone di raw data
                matching_tickets = []
                for ticket_id, phone_info in raw_phones.items():
                    if phone_info['phone'] and complaint_phone in phone_info['phone']:
                        matching_tickets.append(ticket_id)
                
                if matching_tickets:
                    complaint_tickets[complaint_phone] = {
                        'ticket_numbers': matching_tickets,
                        'lead_time_days': complaint_row.get('Lead Time (Solved)'),
                        'complaint_data': complaint_row.to_dict()
                    }
                    print(f"✅ Matched phone {complaint_phone} with tickets: {matching_tickets}")
            
            print(f"📊 Found {len(complaint_tickets)} complaint-ticket matches")
            return complaint_tickets
            
        except Exception as e:
            print(f"❌ Error in complaint matching: {e}")
            return complaint_tickets
        
    def _extract_phones_from_raw_data(self, df):
        """Extract phone numbers dari raw data"""
        phone_info = {}
        
        for ticket_id in df['Ticket Number'].unique():
            ticket_df = df[df['Ticket Number'] == ticket_id]
            
            phone = None
            # Cari phone number di semua kolom teks
            for col in ticket_df.columns:
                if pd.api.types.is_string_dtype(ticket_df[col]):
                    phone_patterns = [r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b', r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b']
                    for pattern in phone_patterns:
                        matches = ticket_df[col].astype(str).str.extract(f'({pattern})', expand=False)
                        if not matches.isna().all():
                            phone = matches.dropna().iloc[0] if not matches.dropna().empty else None
                            if phone:
                                phone = re.sub(r'\D', '', phone)  # Clean phone
                                break
                if phone:
                    break
            
            phone_info[ticket_id] = {'phone': phone}
        
        return phone_info

# Conversation Parser dengan Logic Baru
class ConversationParser:
    def __init__(self):
        self.question_indicators = [
            '?', 'apa', 'bagaimana', 'berapa', 'kapan', 'dimana', 'kenapa',
            'bisa', 'boleh', 'minta', 'tolong', 'tanya', 'info', 'caranya',
            'mau tanya', 'boleh tanya', 'minta info', 'berapa harga',
            'bagaimana cara', 'bisa tolong', 'mohon bantuan', 'gimana'
        ]

        self.problem_indicators = [
            'masalah', 'problem', 'error', 'gagal', 'tidak bisa', 'tidak bisa', 
            'kendala', 'gangguan', 'trouble', 'komplain', 'keluhan', 'kecewa',
            'rusak', 'blank', 'kosong', 'hang', 'lambat', 'eror', 'bug'
        ]
        
        self.operator_greeting_patterns = [
            r"selamat\s+(pagi|siang|sore|malam)",
            r"selamat\s+\w+\s+selamat\s+datang",
            r"selamat\s+datang",
            r"dengan\s+\w+\s+apakah\s+ada",
            r"ada\s+yang\s+bisa\s+dibantu",
            r"boleh\s+dibantu",
            r"bisa\s+dibantu", 
            r"halo.*selamat",
            r"hai.*selamat",
            r"perkenalkan.*saya",
            r"layanan\s+live\s+chat",
            r"live\s+chat\s+toyota",
            r"toyota\s+astra\s+motor"
        ]

        self.vague_patterns = [
            'mau nanya', 'mau tanya', 'boleh tanya', 'saya ingin bertanya', 
            'saya mau tanya', 'halo', 'hai', 'permisi'
        ]

    def _fallback_qa_parsing(self, ticket_df):
        """Fallback parsing method ketika parsing utama gagal - METHOD YANG HILANG"""
        print("   🔄 Using fallback QA parsing...")
        
        conv_df = ticket_df.copy().sort_values('parsed_timestamp').reset_index(drop=True)
        qa_pairs = []
        
        current_question = None
        question_time = None
        
        for idx, row in conv_df.iterrows():
            role = str(row['Role']).lower()
            message = str(row['Message'])
            
            # Customer messages as potential questions
            if 'customer' in role:
                if self._is_potential_question_fallback(message):
                    # Save previous question if exists
                    if current_question:
                        qa_pairs.append({
                            'question': current_question,
                            'question_time': question_time,
                            'is_answered': False,
                            'answer': 'NO_ANSWER',
                            'answer_time': None,
                            'lead_time_minutes': None,
                            'note': 'Unanswered (fallback)'
                        })
                    
                    current_question = message
                    question_time = row['parsed_timestamp']
                    print(f"   💬 Fallback question: {message[:50]}...")
            
            # Operator messages as potential answers
            elif current_question and any(keyword in role for keyword in ['operator', 'agent']):
                # Simple answer assignment
                lead_time = (row['parsed_timestamp'] - question_time).total_seconds()
                
                qa_pairs.append({
                    'question': current_question,
                    'question_time': question_time,
                    'is_answered': True,
                    'answer': message,
                    'answer_time': row['parsed_timestamp'],
                    'lead_time_seconds': lead_time,
                    'lead_time_minutes': round(lead_time / 60, 2),
                    'lead_time_hhmmss': self._seconds_to_hhmmss(lead_time),
                    'note': 'Answered (fallback)'
                })
                
                print(f"   ✅ Fallback answer: {message[:50]}...")
                current_question = None
                question_time = None
        
        # Handle last question if unanswered
        if current_question:
            qa_pairs.append({
                'question': current_question,
                'question_time': question_time,
                'is_answered': False,
                'answer': 'NO_ANSWER',
                'answer_time': None,
                'lead_time_minutes': None,
                'note': 'Unanswered (fallback)'
            })
        
        print(f"   📊 Fallback parsing found {len(qa_pairs)} Q-A pairs")
        return qa_pairs

    def _simple_qa_parsing(self, ticket_df):
        """Simple Q-A parsing untuk cases yang straightforward - METHOD BARU"""
        print("   🔄 Using simple Q-A parsing...")
        
        conv_df = ticket_df.copy().sort_values('parsed_timestamp').reset_index(drop=True)
        qa_pairs = []
        
        current_question = None
        question_time = None
        
        for idx, row in conv_df.iterrows():
            role = str(row['Role']).lower()
            message = str(row['Message'])
            
            # Customer messages
            if any(keyword in role for keyword in ['customer', 'user', 'pelanggan']):
                if self._is_basic_question(message):
                    # Save previous question if exists
                    if current_question:
                        qa_pairs.append(self._create_qa_pair(current_question, question_time, None, None))
                    
                    current_question = message
                    question_time = row['parsed_timestamp']
                    print(f"   💬 Simple question: {message[:50]}...")
            
            # Operator messages
            elif current_question and any(keyword in role for keyword in ['operator', 'agent', 'admin', 'cs']):
                if not self._is_greeting_message(message):
                    lead_time = (row['parsed_timestamp'] - question_time).total_seconds()
                    qa_pairs.append(self._create_qa_pair(current_question, question_time, message, row['parsed_timestamp'], lead_time))
                    print(f"   ✅ Simple answer: {message[:50]}...")
                    
                    current_question = None
                    question_time = None
        
        # Handle last question
        if current_question:
            qa_pairs.append(self._create_qa_pair(current_question, question_time, None, None))
        
        return qa_pairs

    def _is_potential_question_fallback(self, message):
        """Simple question detection untuk fallback parsing"""
        if not message or len(message.strip()) < 5:
            return False
            
        message_lower = message.lower()
        
        # Skip obvious non-questions
        if any(pattern in message_lower for pattern in ['kode undian', 'lucky draw', 'hadiah', 'selamat']):
            return False
            
        # Simple question indicators
        has_question_mark = '?' in message_lower
        has_question_word = any(word in message_lower for word in ['apa', 'bagaimana', 'berapa', 'kapan', 'dimana', 'kenapa'])
        is_long_enough = len(message_lower.split()) >= 3
        
        return (has_question_mark or has_question_word) and is_long_enough
        
        self.vague_patterns = config.VAGUE_QUESTION_PATTERNS

    def _is_basic_question(self, message):
        """Basic question detection - lebih longgar dari meaningful question"""
        if not message or len(message.strip()) < 3:
            return False
            
        message_lower = message.lower()
        
        # Skip obvious non-questions
        if any(pattern in message_lower for pattern in ['kode undian', 'lucky draw', 'selamat pagi', 'selamat siang']):
            return False
            
        # Basic question indicators
        has_question_mark = '?' in message_lower
        has_question_word = any(word in message_lower for word in self.question_indicators)
        is_reasonable_length = len(message_lower.split()) >= 2
        
        return (has_question_mark or has_question_word) and is_reasonable_length

    def parse_conversation(self, ticket_df):
        """Parse conversation dengan multiple fallback strategies"""
        print(f"   🔄 Parsing conversation with {len(ticket_df)} messages")
        
        # Strategy 1: Main parsing
        qa_pairs = self._parse_conversation_main(ticket_df)
        
        if qa_pairs:
            print(f"   ✅ Main parsing successful: {len(qa_pairs)} Q-A pairs")
            return qa_pairs
        
        # Strategy 2: Simple Q-A parsing
        print("   ⚠️ Main parsing failed, trying simple Q-A parsing...")
        qa_pairs = self._simple_qa_parsing(ticket_df)
        
        if qa_pairs:
            print(f"   ✅ Simple parsing successful: {len(qa_pairs)} Q-A pairs")
            return qa_pairs
        
        # Strategy 3: Fallback parsing
        print("   ⚠️ Simple parsing failed, trying fallback parsing...")
        qa_pairs = self._fallback_qa_parsing(ticket_df)
        
        if qa_pairs:
            print(f"   ✅ Fallback parsing successful: {len(qa_pairs)} Q-A pairs")
            return qa_pairs
        
        print("   ❌ All parsing strategies failed")
        return []

    def _parse_conversation_main(self, ticket_df):
        """Main parsing logic - dipindahkan dari parse_conversation"""
        conv_df = ticket_df.copy().sort_values('parsed_timestamp').reset_index(drop=True)
        
        # 🆕 CARI REOPENED TIME DULU
        reopened_time = None
        reopened_idx = None
        for idx, row in conv_df.iterrows():
            if "Ticket Has Been Reopened by" in str(row['Message']):
                reopened_time = row['parsed_timestamp']
                reopened_idx = idx
                print(f"   🔄 REOPENED found at index {idx}, time: {reopened_time}")
                break
        
        qa_pairs = []
        current_question = None
        question_time = None
        question_context = []
        
        for idx, row in conv_df.iterrows():
            role = str(row['Role']).lower()
            message = str(row['Message'])
            timestamp = row['parsed_timestamp']
            
            # CUSTOMER MESSAGE - potential question
            if any(keyword in role for keyword in ['customer', 'user', 'pelanggan']):
                if self._is_problem_statement(message) or self._is_meaningful_question(message):
                    # Jika ada previous question, simpan dulu
                    if current_question:
                        enhanced_question = self._enhance_question_with_context(current_question, question_context)
                        self._save_qa_pair(qa_pairs, enhanced_question, question_time, None, None)
                    
                    # Start new question
                    current_question = message
                    question_time = timestamp
                    question_context = [message]
                    print(f"   💬 Question at {idx}: {message[:50]}...")
            
            # OPERATOR MESSAGE - potential answer
            elif current_question and any(keyword in role for keyword in ['operator', 'agent', 'admin', 'cs']):
                answer = message
                
                if self._is_generic_reply(answer):
                    continue
                
                time_gap = (timestamp - question_time).total_seconds()
                
                if time_gap >= 0:  
                    lead_time = time_gap
                    enhanced_question = self._enhance_question_with_context(current_question, question_context)
                    
                    # 🆕 TANDAI JENIS ANSWER
                    answer_note = ""
                    if reopened_time and timestamp > reopened_time:
                        answer_note = "FINAL_ANSWER"
                        print(f"   🎯 FINAL ANSWER after reopened: {answer[:50]}...")
                    elif reopened_time and timestamp < reopened_time:
                        answer_note = "FIRST_ANSWER"
                        print(f"   🔥 FIRST ANSWER before reopened: {answer[:50]}...")
                    else:
                        answer_note = "NORMAL_ANSWER"
                    
                    self._save_qa_pair(qa_pairs, enhanced_question, question_time, answer, timestamp, role, lead_time, answer_note)
                    print(f"   ✅ Answer at {idx}: {answer[:50]}... (LT: {lead_time/60:.1f}min) - {answer_note}")
                    
                    # 🆕 JANGAN RESET QUESTION untuk serious cases
                    # Biarkan question aktif untuk multiple answers
                    if reopened_time and timestamp < reopened_time:
                        print(f"   ⏳ Waiting for final answer after reopened...")
                        # Tetap pertahankan question untuk final answer
                    else:
                        # Reset hanya jika bukan serious case
                        current_question = None
                        question_time = None
                        question_context = []
        
        # Handle last question
        if current_question:
            enhanced_question = self._enhance_question_with_context(current_question, question_context)
            self._save_qa_pair(qa_pairs, enhanced_question, question_time, None, None)
            print(f"   ❓ Unanswered: {current_question[:50]}...")
        
        qa_pairs = sorted(qa_pairs, key=lambda x: x['question_time'] if x['question_time'] else pd.Timestamp.min)
        
        print(f"   ✅ Found {len(qa_pairs)} Q-A pairs")
        return qa_pairs
        
    def _is_truly_generic_reply(self, message):
        """Skip hanya yang benar-benar generic"""
        message_lower = str(message).lower()
        truly_generic = [
            r'virtual\s+assistant',
            r'akan\s+segera\s+menghubungi', 
            r'dalam\s+antrian',
            r'silakan\s+memilih\s+dari\s+menu',
            r'terima kasih telah menghubungi',
            r'selamat beraktivitas'
        ]
        return any(re.search(pattern, message_lower) for pattern in truly_generic)

    def _is_problem_statement(self, message):
        """Deteksi jika message adalah problem statement, bukan sekedar question"""
        message_lower = message.lower()
        
        problem_indicators = [
            'masalah', 'problem', 'error', 'gagal', 'tidak bisa', 'kendala', 
            'gangguan', 'trouble', 'rusak', 'blank', 'kosong', 'hang', 'lambat'
        ]
        
        has_problem_keyword = any(keyword in message_lower for keyword in problem_indicators)
        is_detailed = len(message_lower.split()) > 8  # Lebih detail dari sekedar greeting
        
        return has_problem_keyword and is_detailed

    def _is_meaningful_customer_message(self, message):
        """Check jika message meaningful - LEBIH FLEXIBLE"""
        if not message or len(message.strip()) < 3:
            return False
            
        message_lower = message.lower().strip()
        
        # Skip very short messages yang cuma greetings
        greetings = ['halo', 'hai', 'hi', 'selamat', 'pagi', 'siang', 'sore', 'malam', 'ok', 'thanks', 'thank you']
        words = message_lower.split()
        
        # Jika hanya 1-2 kata dan itu greetings, skip
        if len(words) <= 2 and any(word in words for word in greetings):
            return False
        
        # 🆕 TERIMA LEBIH BANYAK: Jika mengandung kata tanya atau masalah
        has_question_indicator = any(indicator in message_lower for indicator in self.question_indicators)
        has_problem_indicator = any(indicator in message_lower for indicator in self.problem_indicators)
        has_question_mark = '?' in message_lower
        
        # Terima jika ada salah satu indicator
        return has_question_indicator or has_problem_indicator or has_question_mark or len(words) >= 3
    
    def _is_meaningful_question(self, message):
        """Check jika message meaningful question - DIPERBAIKI dengan vague_patterns"""
        if not message or len(message.strip()) < 3:
            return False
            
        message_lower = message.lower().strip()
        
        # PERBAIKAN: Gunakan self.vague_patterns yang sudah didefinisikan
        if any(pattern in message_lower for pattern in self.vague_patterns):
            # Jika hanya mengandung pattern vague tanpa konten lain, skip
            words = message_lower.split()
            vague_words = sum(1 for word in words if any(pattern in word for pattern in self.vague_patterns))
            if vague_words >= len(words) * 0.5:  # 50% kata adalah vague
                return False
        
        # Skip very short messages yang cuma greetings
        greetings = ['halo', 'hai', 'hi', 'selamat', 'pagi', 'siang', 'sore', 'malam']
        words = message_lower.split()
        if len(words) <= 2 and any(word in words for word in greetings):
            return False
        
        # Question indicators
        has_question_indicator = any(indicator in message_lower for indicator in self.question_indicators)
        has_question_mark = '?' in message_lower
        
        # Meaningful content check
        meaningful_words = [w for w in words if len(w) > 2 and w not in greetings]
        has_meaningful_content = len(meaningful_words) >= 2
        
        # Additional check for problem statements
        if not has_meaningful_content:
            has_meaningful_content = self._is_problem_statement(message)
            
        return (has_question_indicator and has_meaningful_content) or has_question_mark or len(meaningful_words) >= 3
        
    def _is_related_followup(self, new_message, current_question):
        """Cek jika message adalah follow-up dari question sebelumnya"""
        current_lower = current_question.lower()
        new_lower = new_message.lower()
        
        # Cari overlap kata kunci
        current_words = set(current_lower.split())
        new_words = set(new_lower.split())
        
        overlap = len(current_words.intersection(new_words))
        return overlap >= 2  # Minimal 2 kata yang sama
    
    def _enhance_question_with_context(self, main_question, context_messages):
        """Gabungkan context untuk membuat question lebih lengkap"""
        if len(context_messages) <= 1:
            return main_question
        
        # Gabungkan messages yang related
        enhanced = main_question
        for msg in context_messages[1:]:  # Skip yang pertama (sudah main_question)
            if msg != main_question:
                enhanced += f" | {msg}"
        
        return enhanced[:500]  # Batasi panjang
    
    def _is_generic_reply(self, message):
        """Skip generic/bot replies"""
        message_lower = str(message).lower()
        generic_patterns = [
            r'virtual\s+assistant',
            r'akan\s+segera\s+menghubungi', 
            r'dalam\s+antrian',
            r'silakan\s+memilih\s+dari\s+menu'
        ]
        return any(re.search(pattern, message_lower) for pattern in generic_patterns)
    
    def _save_qa_pair(self, qa_pairs, question, question_time, answer, answer_time, answer_role=None, lead_time=None, note=""):
        """Save Q-A pair ke list - ENHANCED VERSION"""
        pair_data = {
            'question': question,
            'question_time': question_time,
            'is_answered': answer is not None
        }
        
        if answer:
            pair_data.update({
                'answer': answer,
                'answer_time': answer_time,
                'answer_role': answer_role,
                'lead_time_seconds': lead_time,
                'lead_time_minutes': round(lead_time / 60, 2) if lead_time else None,
                'lead_time_hhmmss': self._seconds_to_hhmmss(lead_time) if lead_time else None,
                'note': note.strip()
            })
        else:
            pair_data.update({
                'answer': 'NO_ANSWER',
                'answer_time': None,
                'answer_role': None,
                'lead_time_seconds': None,
                'lead_time_minutes': None,
                'lead_time_hhmmss': None,
                'note': 'Unanswered'
            })
        
        qa_pairs.append(pair_data)

    def _is_greeting_message(self, message):
        """Cek jika message adalah greeting"""
        message_lower = message.lower()
        greetings = [
            'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam',
            'halo', 'hai', 'dengan senang hati', 'ada yang bisa dibantu'
        ]
        return any(greeting in message_lower for greeting in greetings)

    def _create_qa_pair(self, question, question_time, answer, answer_time, lead_time=None):
        """Create standardized Q-A pair"""
        pair = {
            'question': question,
            'question_time': question_time,
            'is_answered': answer is not None
        }
        
        if answer:
            pair.update({
                'answer': answer,
                'answer_time': answer_time,
                'lead_time_seconds': lead_time,
                'lead_time_minutes': round(lead_time / 60, 2) if lead_time else None,
                'lead_time_hhmmss': self._seconds_to_hhmmss(lead_time) if lead_time else None,
                'note': 'Simple parsing'
            })
        else:
            pair.update({
                'answer': 'NO_ANSWER',
                'answer_time': None,
                'lead_time_seconds': None,
                'lead_time_minutes': None,
                'lead_time_hhmmss': None,
                'note': 'Unanswered (simple parsing)'
            })
        
        return pair

    def _seconds_to_hhmmss(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except:
            return "00:00:00"
   
    def detect_conversation_start(self, ticket_df):
        """Deteksi kapan conversation benar-benar dimulai dengan operator"""
        ticket_df = ticket_df.sort_values('parsed_timestamp').reset_index(drop=True)
        
        print(f"   🔍 Analyzing {len(ticket_df)} messages for conversation start...")
        
        # Cari operator greeting message
        for idx, row in ticket_df.iterrows():
            message = str(row['Message']).lower()
            role = str(row['Role']).lower()
            
            if any(keyword in role for keyword in ['operator', 'agent', 'admin', 'cs']):
                for pattern in self.operator_greeting_patterns:
                    if re.search(pattern, message, re.IGNORECASE):
                        print(f"   ✅ Conversation start: operator greeting at position {idx}")
                        return row['parsed_timestamp']
        
        # Fallback: first operator message
        for idx, row in ticket_df.iterrows():
            role = str(row['Role']).lower()
            if any(keyword in role for keyword in ['operator', 'agent', 'admin', 'cs']):
                print(f"   ✅ Conversation start: first operator message at position {idx}")
                return row['parsed_timestamp']
        
        print("   ❌ No conversation start detected")
        return None

import re
from datetime import timedelta

import re
from datetime import timedelta

class MainIssueDetector:
    def __init__(self):
        # 1. Keywords Masalah
        self.problem_keywords = [
            'bagaimana', 'mobil saya', 'saya belum dapat', 'saya ingin menanyakan', 
            'stok', 'mobil saya mogok', 'apa bisa', 'masalah', 'problem', 'error', 
            'gagal', 'tidak bisa', 'kendala', 'gangguan', 'trouble', 'komplain', 
            'keluhan', 'kecewa', 'rusak', 'blank', 'kosong', 'hang', 'lambat', 
            'eror', 'bug', 'bantu', 'tolong', 'mohon', 'help'
        ]
        
        # 2. Keywords Urgent
        self.urgent_keywords = [
            'mendesak', 'urgent', 'segera', 'penting', 'cepat', 'sekarang',
            'hari ini', 'besok', 'deadline', 'target', 'asap', 'lekas',
            'secepatnya', 'sekarang juga'
        ]
        
        # 3. Keywords Komplain Keras
        self.complaint_keywords = [
            'komplain', 'kecewa', 'marah', 'protes', 'pengaduan', 'keluhan',
            'sakit hati', 'tidak puas', 'keberatan', 'sangat kecewa', 'anjir', 
            'bangsat', 'bodoh', 'tolol', 'goblok', 'sial', 'jelek', 'buruk'
        ]

        # 4. Pola Vague (Basa-basi)
        self.vague_patterns = [
            'mau nanya', 'mau tanya', 'boleh tanya', 'saya ingin bertanya', 
            'saya mau tanya', 'halo', 'hai', 'permisi', 'pagi', 'siang', 'sore', 
            'malam', 'test', 'tes', 'ping', 'hallo', 'ok', 'oke', 'baik', 'bro', 
            'sis', 'kak', 'admin', 'bang', 'dek', 'mas', 'mbak'
        ]

        # 5. Blacklist Kata-kata Bot/Navigasi
        self.bot_navigation_keywords = [
            'balik ke main menu', 'main menu', 'others', 'pusat bantuan', 
            'hubungi cs', 'setuju', 'tidak setuju', 'ya', 'tidak', 'live chat', 
            'chat dengan agen', 'menu utama', 'kembali', 'pilihan', 'option',
            'terima kasih', 'thanks', 'thank you', 'okay', 'siap', 'roger',
            'baik terima kasih', 'terimakasih', 'sip', 'oke baik'
        ]
        
        # 6. System/Reopening Messages (untuk di-skip)
        self.system_reopen_patterns = [
            r'ticket.*reopen', r'reopened.*by', r'dibuka.*kembali',
            r'dibuka ulang', r'pesan.*tertutup', r'closed.*message', 
            r'auto.*response', r'system.*notification', r'notifikasi.*sistem', 
            r'chat.*ditutup', r'session.*expired', r'time.*out', 
            r'terima kasih.*telah menghubungi', r'terimakasih.*telah.*menghubungi',
            r'percakapan.*ditutup', r'ticket has been reopened',
            r'chat.*reopen', r'percakapan.*dibuka.*kembali',
            r'has been closed', r'telah ditutup', r'this ticket.*closed'
        ]

    def _get_detection_reason(self, score):
        """Mapping score ke alasan"""
        reasons = {
            10: "Urgent complaint detected",
            9: "Serious complaint detected", 
            8: "Urgent problem detected",
            7: "Problem with complaint tone",
            6: "Clear problem statement",
            5: "Problem keyword found",
            4: "Detailed question/query",
            3: "Follow-up question",
            2: "Simple question/greeting",
            1: "Minimal context message",
            0: "No issue detected"
        }
        return reasons.get(score, "Unknown score")
    
    def _is_pure_reopen_message(self, message):
        """Cek apakah pesan HANYA berisi reopening notification"""
        if not message:
            return False
            
        message_lower = str(message).lower().strip()
        
        # Exact matches untuk pesan reopening
        pure_reopen_phrases = [
            'ticket has been reopened by',
            'ticket has been reopened',
            'ticket reopened by',
            'chat has been reopened',
            'percakapan telah dibuka kembali',
            'tiket telah dibuka kembali',
            'dibuka kembali oleh',
            'reopened by',
            'ticket dibuka kembali',
            'this ticket has been reopened'
        ]
        
        for phrase in pure_reopen_phrases:
            if phrase in message_lower:
                return True
        
        # Jika pesan sangat pendek dan mengandung kata kunci reopening
        if len(message_lower.split()) <= 8:
            reopen_keywords = ['reopen', 'reopened', 'dibuka kembali', 'dibuka ulang']
            if any(keyword in message_lower for keyword in reopen_keywords):
                # Tapi pastikan bukan pertanyaan customer
                if not any(q_word in message_lower for q_word in ['?', 'bisa', 'boleh', 'minta', 'tolong']):
                    return True
        
        return False
    
    def _is_system_reopen_message(self, message, role):
        """Cek apakah pesan adalah system/reopen message"""
        if not message:
            return False
            
        message_lower = str(message).lower()
        role_lower = str(role).lower()
        
        # Cek pure reopen message pertama (paling penting!)
        if self._is_pure_reopen_message(message):
            return True
        
        # Cek berdasarkan role
        system_roles = ['system', 'bot', 'auto', 'notification', 'automatic', 'ai', 'chatbot']
        if any(sys_role in role_lower for sys_role in system_roles):
            return True
        
        # Cek EXACT PHRASE untuk system messages
        exact_system_phrases = [
            'ticket has been reopened by',
            'this ticket has been reopened',
            'chat has been closed',
            'percakapan telah ditutup',
            'session expired',
            'time out',
            'terima kasih telah menghubungi',
            'terimakasih telah menghubungi'
        ]
        
        for phrase in exact_system_phrases:
            if phrase in message_lower:
                return True
        
        # Cek berdasarkan content pattern (case-insensitive)
        for pattern in self.system_reopen_patterns:
            try:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return True
            except re.error:
                continue
        
        # Cek jika mengandung kata kunci reopening tanpa konteks pertanyaan
        if len(message_lower.split()) <= 10:
            reopen_keywords = ['reopened', 'dibuka kembali', 're-open']
            message_words = message_lower.split()
            reopen_word_count = 0
            
            for word in message_words:
                for keyword in reopen_keywords:
                    if keyword in word:
                        reopen_word_count += 1
                        break
            
            # Jika sebagian besar kata adalah reopening keywords
            if reopen_word_count > 0 and reopen_word_count >= len(message_words) / 2:
                return True
        
        return False
    
    def _check_if_reopened_ticket(self, ticket_df):
        """Cek apakah ticket mengandung pesan reopening"""
        if ticket_df is None or ticket_df.empty:
            return False
            
        # Cari pesan reopening di dataframe
        for _, row in ticket_df.iterrows():
            message = str(row['Message'])
            role = str(row.get('Role', ''))
            
            if self._is_pure_reopen_message(message):
                return True
                
            # Atau cek dengan metode lengkap
            if self._is_system_reopen_message(message, role):
                # Tapi pastikan ini bukan customer yang bilang "reopen"
                role_lower = role.lower()
                if any(cust_role in role_lower for cust_role in ['customer', 'user', 'client', 'pengguna']):
                    # Customer minta reopen, ini valid issue
                    continue
                return True
                
        return False
    
    def _handle_reopened_ticket(self, qa_pairs, ticket_df):
        """
        Handle khusus untuk reopened tickets:
        1. Cari percakapan ASLI sebelum pesan reopening
        2. Ambil main issue dari percakapan awal
        3. Ignore sistem message setelah reopening
        """
        print("   🔄 Detected REOPENED ticket - using special logic")
        
        # Cari timestamp pesan reopening PERTAMA
        reopening_times = []
        
        for qa in qa_pairs:
            q_lower = qa['question'].lower()
            if self._is_pure_reopen_message(qa['question']):
                reopening_times.append({
                    'time': qa['question_time'],
                    'message': qa['question'],
                    'is_pure': True
                })
            elif self._is_system_reopen_message(qa['question'], ''):
                reopening_times.append({
                    'time': qa['question_time'],
                    'message': qa['question'],
                    'is_pure': False
                })
        
        # Juga cek di ticket_df
        if ticket_df is not None:
            for _, row in ticket_df.iterrows():
                message = str(row['Message'])
                if self._is_pure_reopen_message(message):
                    reopening_times.append({
                        'time': row.get('parsed_timestamp'),
                        'message': message,
                        'is_pure': True,
                        'source': 'ticket_df'
                    })
        
        if reopening_times:
            # Sort by time
            reopening_times.sort(key=lambda x: x['time'] if x['time'] else '')
            
            # Ambil reopening time PERTAMA
            first_reopen = reopening_times[0]
            reopening_time = first_reopen['time']
            reopening_message = first_reopen['message']
            
            print(f"   ⏰ First reopening at: {reopening_time} - {reopening_message[:50]}...")
            
            if reopening_time:
                # Ambil chat SEBELUM reopening
                pre_reopen_pairs = [
                    qa for qa in qa_pairs 
                    if qa['question_time'] and qa['question_time'] < reopening_time
                ]
                
                # Filter bot navigation
                pre_reopen_pairs = [
                    qa for qa in pre_reopen_pairs 
                    if not self._is_bot_navigation(qa['question'])
                ]
                
                if pre_reopen_pairs:
                    print(f"   🔍 Found {len(pre_reopen_pairs)} messages BEFORE reopening")
                    
                    # Coba strict detection dulu
                    main_issue = self._strict_main_issue_detection(pre_reopen_pairs)
                    if main_issue:
                        main_issue['detection_reason'] = f"Reopened ticket - {main_issue['detection_reason']}"
                        main_issue['is_reopened_ticket'] = True
                        main_issue['reopening_time'] = reopening_time
                        return main_issue
                    
                    # Coba relaxed detection
                    main_issue = self._relaxed_main_issue_detection(pre_reopen_pairs)
                    if main_issue:
                        main_issue['detection_reason'] = f"Reopened ticket - {main_issue['detection_reason']}"
                        main_issue['is_reopened_ticket'] = True
                        main_issue['reopening_time'] = reopening_time
                        return main_issue
                    
                    # Kalau masih ga ketemu, ambil yang terakhir sebelum reopening
                    last_pre_reopen = pre_reopen_pairs[-1]
                    return {
                        'question': last_pre_reopen['question'],
                        'question_time': last_pre_reopen['question_time'],
                        'detection_score': 2,
                        'detection_reason': 'Reopened ticket - Last message before reopening',
                        'is_reopened_ticket': True,
                        'reopening_time': reopening_time,
                        'source_type': 'REOPENED_PRE'
                    }
        
        print("   ⚠️ Reopening time not found or no pre-reopen messages, using standard logic")
        return None
    
    def detect_main_issue(self, qa_pairs, ticket_df):
        """
        Logika Utama:
        1. Cek apakah ini reopened ticket → handle khusus
        2. Cari titik mulai Operator (Start Time).
        3. Ambil chat SETELAH operator masuk.
        4. Jika kosong, ambil chat ANTRIAN (15 menit sebelum operator masuk).
        5. Jika operator ada tapi chat customer kosong/invalid → Return None/Greeting.
        """
        if not qa_pairs:
            print("   ❌ No Q-A pairs available")
            return None
        
        print(f"   🔍 Detecting main issue from {len(qa_pairs)} Q-A pairs...")
        
        # --- LANGKAH 0: Cek Reopened Ticket ---
        if ticket_df is not None and not ticket_df.empty:
            if self._check_if_reopened_ticket(ticket_df):
                reopened_result = self._handle_reopened_ticket(qa_pairs, ticket_df)
                if reopened_result:
                    print(f"   ✅ Using pre-reopen conversation as main issue")
                    return reopened_result
                print("   ⚠️ Reopened ticket but no pre-reopen messages found")
        
        # --- LANGKAH 1: Tentukan Waktu Mulai Operator ---
        conversation_start_time = self._find_conversation_start_time(ticket_df)
        
        target_qa_pairs = []
        source_type = "ALL"

        if conversation_start_time:
            print(f"   ⏰ Operator joined at: {conversation_start_time}")
            
            # A. Priority 1: Chat SETELAH Operator Masuk
            post_operator_pairs = [
                qa for qa in qa_pairs 
                if qa['question_time'] and qa['question_time'] >= conversation_start_time
            ]
            
            # Filter sampah bot navigation DAN system messages
            post_operator_pairs = [
                qa for qa in post_operator_pairs 
                if not self._is_bot_navigation(qa['question']) and 
                   not self._is_system_reopen_message(qa['question'], '')
            ]
            
            if post_operator_pairs:
                print(f"   📝 Found {len(post_operator_pairs)} valid messages AFTER operator joined")
                target_qa_pairs = post_operator_pairs
                source_type = "POST_OP"
            else:
                # B. Priority 2: Chat Saat ANTRIAN (Queue Buffer 15 Menit)
                print("   ⚠️ No valid message after operator. Checking queue buffer (15 mins before)...")
                queue_start_time = conversation_start_time - timedelta(minutes=15)
                
                queue_pairs = [
                    qa for qa in qa_pairs
                    if qa['question_time'] and queue_start_time <= qa['question_time'] < conversation_start_time
                ]
                
                # Filter sampah bot & system
                queue_pairs = [
                    qa for qa in queue_pairs 
                    if not self._is_bot_navigation(qa['question']) and 
                       not self._is_system_reopen_message(qa['question'], '')
                ]
                
                if queue_pairs:
                    print(f"   📝 Found {len(queue_pairs)} messages in QUEUE buffer")
                    target_qa_pairs = queue_pairs
                    source_type = "QUEUE"
                else:
                    # C. STOP! Jangan mundur ke bot awal.
                    print("   ❌ Customer silent in Queue & Active session. Marking as unresponsive.")
                    return None 
        else:
            # D. Fallback: Tidak ada operator terdeteksi (Full Bot / Failed Handover)
            print("   ⚠️ No operator detected. Scanning all pairs excluding bot/system commands...")
            target_qa_pairs = [
                qa for qa in qa_pairs 
                if not self._is_bot_navigation(qa['question']) and 
                   not self._is_system_reopen_message(qa['question'], '')
            ]
            source_type = "NO_OP"
        
        if not target_qa_pairs:
            print("   ❌ No valid Q-A pairs after filtering")
            return None

        # --- LANGKAH 2: Deteksi Masalah dari target_qa_pairs ---
        
        # Strategy 1: Strict (Ada keyword masalah jelas)
        main_issue = self._strict_main_issue_detection(target_qa_pairs)
        if main_issue:
            main_issue['source_type'] = source_type
            return main_issue
        
        # Strategy 2: Relaxed (Pertanyaan umum)
        print("   ⚠️ Strict detection failed, trying relaxed detection...")
        main_issue = self._relaxed_main_issue_detection(target_qa_pairs)
        if main_issue:
            main_issue['source_type'] = source_type
            return main_issue
        
        # Strategy 3: Contextual Fallback (Khusus jika sumbernya dari Queue/Post-Op)
        if source_type in ["POST_OP", "QUEUE"]:
            print(f"   ⚠️ Using contextual fallback from {source_type}...")
            # Ambil pesan terakhir customer yang bukan bot command
            for qa in reversed(target_qa_pairs):
                if qa['question'] and len(str(qa['question']).strip()) > 2:
                    return {
                        'question': qa['question'],
                        'question_time': qa['question_time'],
                        'detection_score': 2,
                        'detection_reason': f'Customer message from {source_type}',
                        'source_type': source_type
                    }
        
        print("   ❌ All detection strategies failed")
        return None

    def _find_conversation_start_time(self, ticket_df):
        """Cari kapan percakapan benar-benar dimulai dengan operator (SKIP system messages)"""
        if ticket_df is None or ticket_df.empty:
            return None
            
        ticket_df = ticket_df.sort_values('parsed_timestamp').reset_index(drop=True)
        
        operator_greeting_patterns = [
            r"selamat\s+(pagi|siang|sore|malam).+selamat\s+datang", 
            r"selamat\s+datang\s+di\s+layanan",
            r"dengan\s+\w+\s*,?\s*apakah\s+ada",
            r"ada\s+yang\s+bisa\s+dibantu",
            r"perkenalkan.*nama\s+saya",
            r"saat\s+ini\s+anda\s+terhubung",
            r"terhubung\s+dengan\s+live\s+chat",
            r"halo.*selamat\s+(pagi|siang|sore|malam)",
            r"selamat\s+(pagi|siang|sore|malam).*nama\s+saya",
            r"hai.*selamat\s+(pagi|siang|sore|malam)",
            r"selamat.*(pagi|siang|sore|malam).*saya.*(operator|agent|cs|admin)"
        ]
        
        # Langsung scan dataframe, skip system messages
        for idx, row in ticket_df.iterrows():
            role = str(row.get('Role', '')).lower()
            message = str(row['Message'])
            
            # SKIP system/reopen messages
            if self._is_system_reopen_message(message, role):
                continue
            
            # Cek apakah pengirim adalah manusia/agen
            is_agent = any(k in role for k in ['operator', 'agent', 'admin', 'cs', 'staff', 'support', 'representative', 'officer'])
            
            if is_agent:
                # Cek greeting pattern
                for pattern in operator_greeting_patterns:
                    if re.search(pattern, message, re.IGNORECASE):
                        # Pastikan ini bukan pesan closing
                        if not self._is_closing_message(message):
                            print(f"   ✅ Found operator greeting (non-system): {message[:50]}...")
                            return row['parsed_timestamp']
    
        # Fallback 1: Pesan pertama dari role Operator apapun (tapi skip sistem)
        for idx, row in ticket_df.iterrows():
            role = str(row.get('Role', '')).lower()
            message = str(row['Message'])
            
            # Skip sistem
            if self._is_system_reopen_message(message, role):
                continue
            
            if any(k in role for k in ['operator', 'agent', 'admin', 'cs', 'staff', 'support']):
                # Pastikan bukan closing message
                if not self._is_closing_message(message):
                    print(f"   ⚠️ Using first non-system operator message as anchor: {row['Message'][:30]}...")
                    return row['parsed_timestamp']
    
        # Fallback 2: Jika tidak ditemukan operator di non-system messages
        print("   ⚠️ No operator found in non-system messages, checking all messages...")
        for idx, row in ticket_df.iterrows():
            role = str(row.get('Role', '')).lower()
            message = str(row['Message'])
            
            # Skip jika jelas-jelas PURE system message (tapi allow ambiguous)
            if self._is_pure_reopen_message(message):
                continue
                
            if any(k in role for k in ['operator', 'agent', 'admin', 'cs', 'staff', 'support']):
                if not self._is_closing_message(message):
                    print(f"   ⚠️ Using first available operator message: {row['Message'][:30]}...")
                    return row['parsed_timestamp']
                
        print("   ❌ No valid operator start time found")
        return None

    def _is_closing_message(self, message):
        """Cek apakah pesan adalah closing/ending message"""
        if not message:
            return False
            
        message_lower = str(message).lower()
        
        # Exact closing phrases
        closing_phrases = [
            'terima kasih',
            'terimakasih',
            'thank you',
            'thanks',
            'selesai',
            'finished',
            'tutup percakapan',
            'close chat',
            'percakapan selesai',
            'sampai jumpa',
            'goodbye',
            'have a nice day',
            'semoga harimu menyenangkan',
            'percakapan ini ditutup'
        ]
        
        for phrase in closing_phrases:
            if phrase in message_lower:
                return True
                
        # Pattern untuk closing
        closing_patterns = [
            r'terima kasih.*telah.*menghubungi',
            r'thank you.*for.*contacting',
            r'percakapan.*telah.*ditutup',
            r'chat.*has.*been.*closed'
        ]
        
        for pattern in closing_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return True
                
        return False

    def _is_bot_navigation(self, text):
        """Cek apakah teks adalah command navigasi bot"""
        if not text or str(text).strip() == '':
            return True
            
        text_lower = str(text).lower().strip()
        
        # Skip jika terlalu pendek
        if len(text_lower) < 2:
            return True
        
        # Exact match untuk common bot responses
        exact_bot_responses = [
            'ya', 'tidak', 'ok', 'oke', 'baik', 'sip', 'y', 'n',
            'yes', 'no', 'okay', 'got it', 'roger', 'terima kasih',
            'thanks', 'thank you'
        ]
        
        if text_lower in exact_bot_responses:
            return True
        
        # Cek contain keyword bot
        for keyword in self.bot_navigation_keywords:
            if keyword == text_lower or keyword in text_lower:
                # Tapi jangan flag jika itu bagian dari kalimat panjang
                if len(text_lower.split()) <= 3 or len(text_lower) < 20:
                    return True
        
        # Cek jika hanya berisi angka atau simbol
        if re.match(r'^[\d\s\W]+$', text_lower):
            return True
            
        return False

    def _calculate_question_score(self, question):
        """Hitung skor untuk pertanyaan"""
        if not question:
            return 0
            
        score = 0
        question_lower = str(question).lower()
        words = question_lower.split()
        
        # Base score berdasarkan panjang
        if len(words) >= 20: score += 4
        elif len(words) >= 15: score += 3
        elif len(words) >= 10: score += 2
        elif len(words) >= 5: score += 1
        
        # Keywords scoring (prioritas tinggi)
        if any(k in question_lower for k in self.complaint_keywords):
            score += 5
            # Extra point untuk complaint keras
            hard_complaints = ['anjir', 'bangsat', 'bodoh', 'tolol', 'goblok', 'sial']
            if any(hc in question_lower for hc in hard_complaints):
                score += 2
        
        if any(k in question_lower for k in self.urgent_keywords):
            score += 4
            
        if any(k in question_lower for k in self.problem_keywords):
            score += 3
        
        # Question marks dan tanda tanya
        if '?' in question_lower:
            score += 2
        elif any(word in question_lower for word in ['kenapa', 'mengapa', 'bagaimana', 'kapan', 'dimana', 'berapa']):
            score += 1
        
        # Context indicators
        if any(word in question_lower for word in ['mohon', 'tolong', 'bisa tolong', 'minta tolong']):
            score += 1
            
        # Negative indicators (kurangi score)
        if any(vague in question_lower for vague in self.vague_patterns):
            if len(words) < 5:  # Hanya vague pendek
                score -= 1
        
        # Minimal score 0
        return max(score, 0)

    def _strict_main_issue_detection(self, qa_pairs):
        """Hanya return jika ada score tinggi (masalah jelas)"""
        candidates = []
        for qa in qa_pairs:
            q_text = qa['question']
            
            # Skip yang terlalu pendek untuk strict detection
            if len(str(q_text).split()) < 4: 
                continue
            
            score = self._calculate_question_score(q_text)
            
            # Hanya pertanyaan dengan score tinggi
            if score >= 5:  # Minimal problem dengan detail
                candidates.append((qa, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_qa, best_score = candidates[0]
            
            # Adjust reason berdasarkan score
            if best_score >= 10:
                reason = "Urgent complaint detected"
            elif best_score >= 8:
                reason = "Serious problem detected"
            elif best_score >= 6:
                reason = "Clear problem statement"
            else:
                reason = self._get_detection_reason(best_score)
                
            return {
                'question': best_qa['question'],
                'question_time': best_qa['question_time'],
                'detection_score': best_score,
                'detection_reason': reason
            }
        return None

    def _relaxed_main_issue_detection(self, qa_pairs):
        """Return pertanyaan terbaik yang masuk akal"""
        candidates = []
        for qa in qa_pairs:
            q_text = qa['question']
            
            # Skip yang terlalu singkat atau kosong
            if not q_text or len(str(q_text).strip()) < 3:
                continue
            
            # Skip bot navigation
            if self._is_bot_navigation(q_text):
                continue
            
            score = self._calculate_question_score(q_text)
            candidates.append((qa, score))
            
        if candidates:
            # Sort by score, then by length (prioritize detailed questions)
            candidates.sort(key=lambda x: (x[1], len(str(x[0]['question']).split())), reverse=True)
            best_qa, best_score = candidates[0]
            
            # Update score jadi minimal 1
            final_score = max(best_score, 1)
            
            return {
                'question': best_qa['question'],
                'question_time': best_qa['question_time'],
                'detection_score': final_score,
                'detection_reason': self._get_detection_reason(final_score)
            }
        return None

    def debug_conversation_flow(self, qa_pairs, ticket_df):
        """Function untuk debugging flow deteksi"""
        print("\n" + "="*80)
        print("DEBUG MAIN ISSUE DETECTION")
        print("="*80)
        
        print(f"Total QA Pairs: {len(qa_pairs)}")
        
        # Check for reopening
        is_reopened = self._check_if_reopened_ticket(ticket_df)
        print(f"Reopened Ticket: {is_reopened}")
        
        # Find operator start time
        op_start = self._find_conversation_start_time(ticket_df)
        print(f"Operator Start Time: {op_start}")
        
        # Check system messages
        print("\nSystem/Reopen Messages Detection:")
        if ticket_df is not None:
            for i, row in ticket_df.iterrows():
                msg = str(row['Message'])
                role = str(row.get('Role', ''))
                if self._is_system_reopen_message(msg, role):
                    print(f"  [{i}] SYSTEM: {msg[:80]}...")
        
        # Show all QA pairs dengan scoring
        print("\nQA Pairs with Scoring:")
        for i, qa in enumerate(qa_pairs):
            q_time = qa.get('question_time', 'N/A')
            q_text = str(qa['question'])
            if len(q_text) > 60:
                q_text = q_text[:57] + "..."
                
            score = self._calculate_question_score(qa['question'])
            is_bot = self._is_bot_navigation(qa['question'])
            is_system = self._is_system_reopen_message(qa['question'], '')
            
            flags = []
            if is_bot: flags.append("BOT")
            if is_system: flags.append("SYS")
            if self._is_pure_reopen_message(qa['question']): flags.append("PURE_REOPEN")
            
            flag_str = f"[{','.join(flags)}]" if flags else ""
            
            print(f"  [{i:2d}] {q_time} | Score: {score:2d} {flag_str:15} | Q: {q_text}")
        
        # Run detection
        print("\n" + "-"*80)
        print("RUNNING DETECTION...")
        result = self.detect_main_issue(qa_pairs, ticket_df)
        
        print("\n" + "="*80)
        print("DETECTION RESULT:")
        if result:
            print(f"  ✓ Question: {result['question']}")
            print(f"  ✓ Time: {result['question_time']}")
            print(f"  ✓ Score: {result['detection_score']}")
            print(f"  ✓ Reason: {result['detection_reason']}")
            print(f"  ✓ Source: {result.get('source_type', 'N/A')}")
            if result.get('is_reopened_ticket'):
                print(f"  ✓ Reopened Ticket: Yes (at {result.get('reopening_time')})")
        else:
            print("  ❌ NO MAIN ISSUE DETECTED")
        print("="*80 + "\n")
        
        return result

# Helper function untuk testing
def test_main_issue_detector():
    """Test function untuk verifikasi MainIssueDetector"""
    detector = MainIssueDetector()
    
    # Test 1: Pure reopen message
    test_messages = [
        "Ticket Has Been Reopened by System",
        "ticket has been reopened by admin",
        "This ticket has been reopened",
        "Chat telah dibuka kembali",
        "bisa tolong reopen ticket saya?",
        "mohon dibuka kembali chatnya"
    ]
    
    print("Testing Reopen Message Detection:")
    for msg in test_messages:
        is_pure = detector._is_pure_reopen_message(msg)
        is_system = detector._is_system_reopen_message(msg, 'system')
        print(f"  '{msg[:30]}...' -> Pure: {is_pure}, System: {is_system}")
    
    return detector    
class ReplyAnalyzer:
    def __init__(self, complaint_tickets=None):
        self.complaint_tickets = complaint_tickets or {}
        self.action_keywords = config.ACTION_KEYWORDS
        
        # Extended solution keywords untuk menangkap lebih banyak pattern
        self.solution_keywords_extended = config.SOLUTION_KEYWORDS + [
            'bisa menghubungi', 'silakan menghubungi', 'dapat menghubungi', 'hubungi',
            'nomor', 'telepon', 'tlp', 'call', 'contact', 'kontak',
            'alamat', 'lokasi', 'datang ke', 'kunjungi',
            'website', 'situs', 'online', 'aplikasi',
            'bisa dilihat', 'silakan dilihat', 'dapat dilihat',
            'info', 'informasi', 'detail', 'rincian',
            'proses', 'cara', 'langkah', 'tahap',
            'dijawab', 'diberikan', 'diberitahu', 'dijelaskan'
        ]
        
        # Pola System Log yang HARUS diskip
        self.system_patterns = [
            r"reassigned to",
            r"claimed by system",
            r"ticket has been reopened",
            r"terhubung dengan",
            r"saat ini anda terhubung",
            r"antrian pelayanan"
        ]

        # Pola Basa-basi (Greeting kosong) yang HARUS diskip untuk Complaint
        self.greeting_only_patterns = [
            r"^(halo|hai|pagi|siang|sore|malam).{0,20}$",  # Halo pendek
            r"^ada yang bisa (saya|kami|bantu).{0,30}$",    # Penawaran bantuan generik
            r"^dengan (bapak|ibu).{0,20}$",                 # Konfirmasi nama pendek
            r"^apakah (bapak|ibu) (masih|ada).{0,20}$"      # Cek kehadiran
        ]

        # Keywords yang menandakan Respon Bagus (Empati/Action)
        self.strong_start_keywords = [
            'maaf', 'ketidaknyamanan', 'terkait', 'informasi', 'laporan', 
            'bantu buatkan', 'kendala', 'keluhan', 'stnk', 'bpkb', 'sampaikan',
            'mohon menunggu', 'koordinasi', 'penjelasan'
        ]

        # --- ML SOLUTION MODEL INTEGRATION ---
        self.ml_ready = False
        try:
            # Load model yang ditraining khusus untuk jawaban
            self.vectorizer = joblib.load('models/answer_tfidf.pkl')
            self.model = joblib.load('models/answer_model.pkl')
            self.ml_ready = True
            print("   🤖 ML Solution Model Loaded: Siap memvalidasi jawaban operator.")
        except Exception as e:
            print(f"   ⚠️ ML Solution Model missing ({e}). Running Rule-Based only.")
        
    def _check_enhanced_customer_leave(self, ticket_df, first_reply_found, final_reply_found, question_time):
        """Enhanced customer leave detection yang bisa handle multiple leaves"""
        
        print(f"\n   🔍 Checking customer leave (general)...")
        
        # Sort timeline
        sorted_df = ticket_df.sort_values('parsed_timestamp')
        
        # Cari SEMUA leave messages
        all_leave_messages = sorted_df[
            sorted_df['Message'].str.contains(config.CUSTOMER_LEAVE_KEYWORD, na=False)
        ]
        
        print(f"   📊 Found {len(all_leave_messages)} leave messages total")
        
        if all_leave_messages.empty:
            return False
        
        # Analisis setiap leave message
        for i, (_, leave_row) in enumerate(all_leave_messages.iterrows()):
            leave_time = leave_row['parsed_timestamp']
            leave_text = leave_row['Message']
            
            print(f"\n   🚪 Analyzing leave message #{i+1} at {leave_time}")
            
            # Cari operator message TERAKHIR sebelum leave ini
            operator_messages = sorted_df[
                (sorted_df['parsed_timestamp'] < leave_time) &
                (sorted_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
            ]
            
            if operator_messages.empty:
                print(f"   ⚠️ No operator messages before this leave")
                continue
            
            last_op = operator_messages.iloc[-1]
            last_op_time = last_op['parsed_timestamp']
            
            # Cek customer responses SETELAH operator terakhir
            customer_responses = sorted_df[
                (sorted_df['parsed_timestamp'] > last_op_time) &
                (sorted_df['parsed_timestamp'] < leave_time) &
                (sorted_df['Role'].str.lower().str.contains('customer', na=False))
            ]
            
            print(f"   👤 Customer responses after last operator: {len(customer_responses)}")
            
            # Ini customer leave jika tidak ada customer response
            if len(customer_responses) == 0 and "tidak ada respon" in leave_text.lower():
                print(f"   🎯 CONFIRMED: This is a TRUE customer leave")
                
                # Cek apakah ini leave SETELAH reopened
                reopened_before = self._find_ticket_reopened_time_before(ticket_df, leave_time)
                if reopened_before:
                    print(f"   🔄 This leave occurred AFTER a reopened event")
                
                return True
        
        print(f"   ❌ No valid customer leave found in any of the leave messages")
        return False

    def _find_operator_greetings_before_time(self, ticket_df, target_time):
        """Cari operator greetings sebelum waktu tertentu"""
        operator_messages = ticket_df[
            (ticket_df['parsed_timestamp'] < target_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ]
        
        greetings = []
        for _, msg in operator_messages.iterrows():
            if self._is_operator_greeting(msg['Message']):
                greetings.append(msg)
        
        # PERBAIKAN: Return DataFrame yang benar
        return pd.DataFrame(greetings) if greetings else pd.DataFrame()

    def _is_operator_greeting(self, message):
        """Cek apakah message adalah operator greeting"""
        message_lower = str(message).lower()
        
        greeting_patterns = [
            r"selamat\s+(pagi|siang|sore|malam)",
            r"selamat\s+datang",
            r"dengan\s+\w+\s+apakah\s+ada",
            r"ada\s+yang\s+bisa\s+dibantu",
            r"boleh\s+dibantu",
            r"bisa\s+dibantu",
            r"halo.*selamat",
            r"hai.*selamat",
            r"perkenalkan.*saya",
            r"layanan\s+live\s+chat",
            r"live\s+chat\s+toyota",
            r"toyota\s+astra\s+motor"
        ]
        
        return any(re.search(pattern, message_lower, re.IGNORECASE) for pattern in greeting_patterns)

    def _find_customer_interactions_after_greeting(self, ticket_df, greeting_time, leave_time):
        """Cari interaksi customer setelah greeting dan sebelum leave"""
        customer_messages = ticket_df[
            (ticket_df['parsed_timestamp'] > greeting_time) &
            (ticket_df['parsed_timestamp'] < leave_time) &
            (ticket_df['Role'].str.lower().str.contains('customer', na=False))
        ]
        
        # Filter hanya messages yang meaningful (bukan ucapan singkat)
        meaningful_interactions = []
        for _, msg in customer_messages.iterrows():
            if self._is_meaningful_customer_message(msg['Message']):
                meaningful_interactions.append(msg)
        
        return meaningful_interactions

    def _find_operator_interactions_after_greeting(self, ticket_df, greeting_time, leave_time):
        """Cari interaksi operator meaningful setelah greeting dan sebelum leave"""
        operator_messages = ticket_df[
            (ticket_df['parsed_timestamp'] > greeting_time) &
            (ticket_df['parsed_timestamp'] < leave_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ]
        
        # Filter hanya messages yang meaningful (bukan greeting ulang atau konfirmasi)
        meaningful_interactions = []
        for _, msg in operator_messages.iterrows():
            if self._is_meaningful_operator_message(msg['Message']):
                meaningful_interactions.append(msg)
        
        return meaningful_interactions

    def _is_meaningful_customer_message(self, message):
        """Cek apakah customer message meaningful (bukan ucapan singkat)"""
        if not message or len(message.strip()) < 10:
            return False
            
        message_lower = message.lower().strip()
        
        # Skip very short responses
        short_responses = ['ok', 'oke', 'baik', 'sip', 'terima kasih', 'thanks', 'thank you']
        if message_lower in short_responses:
            return False
            
        # Skip single word responses unless they're questions
        words = message_lower.split()
        if len(words) <= 2 and not any(q in message_lower for q in ['?', 'apa', 'bagaimana', 'berapa']):
            return False
            
        return True

    def _is_meaningful_operator_message(self, message):
        """Cek apakah operator message meaningful (bukan greeting ulang atau konfirmasi)"""
        if not message or len(message.strip()) < 15:
            return False
            
        message_lower = message.lower().strip()
        
        # Skip duplicate greetings
        if self._is_operator_greeting(message):
            return False
            
        # Skip simple confirmations
        simple_confirmations = [
            'baik', 'baik sekali', 'terima kasih', 'silahkan', 'oke', 'ok',
            'baik [a-zA-Z]+', 'terimakasih', 'siap', 'baik ditunggu'
        ]
        
        if any(re.search(pattern, message_lower) for pattern in simple_confirmations):
            return len(message_lower.split()) > 3  # Only if it's more than just the confirmation
            
        # Skip questions about presence
        presence_questions = [
            'apakah masih ada', 'apakah bapak/ibu masih', 'apakah anda masih',
            'apakah masih bersama kami', 'apakah masih online'
        ]
        
        if any(q in message_lower for q in presence_questions):
            return False
            
        return True

    def _is_complaint_ticket(self, ticket_id):
        """Cek apakah ticket termasuk complaint"""
        if not self.complaint_tickets:
            return False, None
            
        for phone, complaint_info in self.complaint_tickets.items():
            if ticket_id in complaint_info.get('ticket_numbers', []):
                return True, complaint_info
        return False, None
        
    def _is_system_message(self, message):
        """Cek apakah pesan adalah log sistem"""
        msg_lower = str(message).lower()
        return any(re.search(p, msg_lower) for p in self.system_patterns)

    def _is_weak_greeting(self, message):
        """Cek apakah pesan hanya greeting basa-basi"""
        msg_lower = str(message).lower().strip()
        # Jika mengandung keyword kuat, bukan weak greeting
        if any(k in msg_lower for k in self.strong_start_keywords):
            return False
        # Cek pattern greeting pendek
        if len(msg_lower.split()) < 6: # Terlalu pendek
            return True
        return any(re.search(p, msg_lower) for p in self.greeting_only_patterns)

    def _analyze_complaint_replies(self, ticket_id, ticket_df, qa_pairs, main_issue, complaint_data):
        """Analyze replies KHUSUS untuk complaint tickets dengan logika filter baru"""
        print("   📋 Analyzing COMPLAINT replies...")
        
        # GUNAKAN METHOD BARU YANG LEBIH PINTAR
        first_reply = self._find_meaningful_complaint_reply(ticket_df, main_issue['question_time'])
        
        first_reply_found = first_reply is not None
        final_reply_found = True  # complaint selalu dianggap ada final reply (dari data complaint)
        
        customer_leave = self._check_enhanced_customer_leave(
            ticket_df, first_reply_found, final_reply_found, main_issue['question_time']
        )
        
        analysis_result = {
            'issue_type': 'complaint',
            'first_reply': first_reply,
            'final_reply': {
                'message': 'Check Complaint Report for More Information',
                'timestamp': None,
                'lead_time_minutes': None,
                'lead_time_days': complaint_data.get('lead_time_days'),
                'note': 'Final resolution from complaint system'
            },
            'customer_leave': customer_leave,
            'requirement_compliant': first_reply_found
        }
        
        print(f"   📊 Complaint analysis result:")
        if first_reply:
            print(f"      ✅ First reply found: {first_reply['message'][:50]}...")
        else:
            print(f"      ❌ First reply Missing")
        
        return analysis_result

    def _find_meaningful_complaint_reply(self, ticket_df, question_time):
        """
        Mencari first reply terbaik untuk komplain.
        Prioritas:
        1. Pesan yang mengandung Empati/Action (Maaf, Laporan, Terkait).
        2. Pesan panjang (> 10 kata) yang bukan system log.
        3. Fallback: Pesan operator pertama yang bukan system log (jika tidak ada yang bagus).
        """
        operator_messages = ticket_df[
            (ticket_df['parsed_timestamp'] > question_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ].sort_values('parsed_timestamp')
        
        # PERBAIKAN: Gunakan .empty untuk cek DataFrame
        if operator_messages.empty:
            return None

        candidates = []
        
        # Pass 1: Filter System Log & Kumpulkan Kandidat
        for _, msg in operator_messages.iterrows():
            message_text = str(msg['Message'])
            
            # Skip System Log mutlak
            if self._is_system_message(message_text):
                continue
                
            lt = (msg['parsed_timestamp'] - question_time).total_seconds()
            
            candidates.append({
                'message': message_text,
                'timestamp': msg['parsed_timestamp'],
                'lead_time_seconds': lt,
                'lead_time_minutes': round(lt/60, 2),
                'lead_time_hhmmss': self._seconds_to_hhmmss(lt),
                'word_count': len(message_text.split()),
                'has_strong_keyword': any(k in message_text.lower() for k in self.strong_start_keywords),
                'is_weak': self._is_weak_greeting(message_text)
            })

        # Pass 2: Cari "The One" (Respon Terbaik)
        
        # Priority A: Ada Keyword Kuat (Maaf, Laporan, Terkait) DAN Bukan Weak Greeting
        for cand in candidates:
            if cand['has_strong_keyword']:
                return cand
        
        # Priority B: Cukup Panjang (> 10 kata) DAN Bukan Weak Greeting
        for cand in candidates:
            if cand['word_count'] > 10 and not cand['is_weak']:
                return cand

        # Priority C: Fallback (Ambil yang pertama yang BUKAN weak greeting)
        for cand in candidates:
            if not cand['is_weak']:
                return cand
                
        # Priority D: Last Resort (Ambil pesan manusia pertama apapun isinya, asal bukan system)
        if candidates:
            return candidates[0]
            
        return None

    def _analyze_normal_replies(self, ticket_df, qa_pairs, main_issue):
        """Analyze replies untuk normal tickets - DIPERBAIKI dengan logic yang lebih baik"""
        print("   📋 Analyzing NORMAL replies...")
        
        # PERBAIKAN: Gunakan method yang lebih baik untuk mencari final reply
        final_reply = self._find_enhanced_solution_reply(ticket_df, main_issue['question_time'])
        
        final_reply_found = final_reply is not None
        first_reply_found = False  # Untuk normal ticket, tidak ada first reply requirement
        
        customer_leave = self._check_enhanced_customer_leave(
            ticket_df, first_reply_found, final_reply_found, main_issue['question_time']
        )
        
        analysis_result = {
            'issue_type': 'normal',
            'first_reply': None,
            'final_reply': final_reply,
            'customer_leave': customer_leave,
            'requirement_compliant': final_reply_found
        }
        
        print(f"   📊 Normal analysis: Final={final_reply_found}, Leave={customer_leave}")
        
        # DEBUG: Print jika tidak ditemukan final reply
        if not final_reply_found:
            print("   🔍 DEBUG: No final reply found. Checking operator messages...")
            operator_messages = ticket_df[
                (ticket_df['parsed_timestamp'] > main_issue['question_time']) &
                (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
            ].sort_values('parsed_timestamp')
            
            if not operator_messages.empty:
                print(f"   🔍 DEBUG: Found {len(operator_messages)} operator messages after question")
                for idx, msg in operator_messages.iterrows():
                    print(f"   🔍 DEBUG Operator msg: {msg['Message'][:100]}...")
                    print(f"   🔍 DEBUG Contains solution: {self._contains_solution_keyword_extended(msg['Message'])}")
                    print(f"   🔍 DEBUG Is closing pattern: {any(c in msg['Message'].lower() for c in config.CLOSING_PATTERNS)}")
        
        return analysis_result

    def _get_ml_solution_similarity(self, text):
        """Cek kemiripan balasan operator dengan database kunci jawaban"""
        if not self.ml_ready or not text or len(text.split()) < 5:
            return 0
        try:
            vector = self.vectorizer.transform([text])
            distances, _ = self.model.kneighbors(vector)
            similarity = 1 - distances[0][0] # Convert distance to similarity
            return similarity
        except:
            return 0

    def _find_enhanced_solution_reply(self, ticket_df, question_time):
            """
            Enhanced method: Gabungan Rule-Based Keyword + ML Semantic Matching
            """
            operator_messages = ticket_df[
                (ticket_df['parsed_timestamp'] > question_time) &
                (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
            ].sort_values('parsed_timestamp')
            
            if operator_messages.empty:
                return None
            
            solution_candidates = []
            
            for _, msg in operator_messages.iterrows():
                message_text = str(msg['Message'])
                message_lower = message_text.lower()
                
                # Filter sampah dasar (System log, closing, greeting pendek)
                if (self._is_system_message(message_text) or 
                    any(c in message_lower for c in config.CLOSING_PATTERNS) or
                    self._is_weak_greeting(message_text)):
                    continue
                
                # --- HYBRID DETECTION LOGIC ---
                
                # 1. Cek Rule-Based (Keyword)
                has_keyword = self._contains_solution_keyword_extended(message_text)
                
                # 2. Cek ML Based (Semantic Similarity)
                ml_score = self._get_ml_solution_similarity(message_text)
                is_ml_match = ml_score > 0.65  # Threshold: 65% mirip dengan kunci jawaban
                
                if is_ml_match:
                    print(f"      🤖 ML Solution Match! (Sim: {ml_score:.2f}) -> '{message_text[:30]}...'")

                # Kriteria Penerimaan: Ada Keyword ATAU Mirip Kunci Jawaban (ML)
                # Syarat tambahan: Panjang minimal tetap berlaku untuk menghindari false positive pendek
                is_valid_candidate = (has_keyword or is_ml_match) and len(message_text.split()) >= 5
                
                if is_valid_candidate:
                    lt = (msg['parsed_timestamp'] - question_time).total_seconds()
                    
                    # Tentukan catatan deteksinya
                    detection_note = 'Rule-based keyword'
                    if is_ml_match and has_keyword: detection_note = 'Strong Match (Keyword + ML)'
                    elif is_ml_match: detection_note = 'ML Semantic Match (No keyword)'
                    
                    solution_candidates.append({
                        'message': message_text,
                        'timestamp': msg['parsed_timestamp'],
                        'lead_time_seconds': lt,
                        'lead_time_minutes': round(lt/60, 2),
                        'lead_time_hhmmss': self._seconds_to_hhmmss(lt),
                        'note': detection_note,
                        'score': ml_score if is_ml_match else 0.5 # Prioritaskan ML match saat sorting
                    })
            
            # Ambil kandidat terbaik
            if solution_candidates:
                # Sort berdasarkan score ML tertinggi, lalu waktu tercepat
                solution_candidates.sort(key=lambda x: (-x['score'], x['lead_time_seconds']))
                return solution_candidates[0]
            
            return None

    def _contains_solution_keyword_extended(self, message):
        """Cek solution keyword dengan daftar yang diperluas"""
        msg = str(message).lower()
        return any(k in msg for k in self.solution_keywords_extended)

    def analyze_replies(self, ticket_id, ticket_df, qa_pairs, main_issue):
        print(f"🔍 Analyzing replies for ticket {ticket_id}")
        
        # DEBUG: Tampilkan timeline lengkap
        print(f"   📅 TICKET TIMELINE DEBUG:")
        sorted_df = ticket_df.sort_values('parsed_timestamp')
        
        # Cari semua event penting
        reopened_times = []
        leave_messages = []
        
        for idx, row in sorted_df.iterrows():
            msg = str(row['Message'])
            role = row['Role']
            time = row['parsed_timestamp']
            
            if "Ticket Has Been Reopened" in msg:
                reopened_times.append(time)
                print(f"   🔄 REOPENED at {time}")
            
            if config.CUSTOMER_LEAVE_KEYWORD in msg:
                leave_messages.append((time, msg))
                print(f"   🚪 LEAVE MESSAGE at {time}")
        
        print(f"   📊 Total reopened events: {len(reopened_times)}")
        print(f"   📊 Total leave messages: {len(leave_messages)}")
        
        # Tampilkan timeline
        print(f"   📋 Complete timeline:")
        for idx, row in sorted_df.iterrows():
            time = row['parsed_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            role = row['Role'][:15]
            msg = row['Message'][:60] + '...' if len(row['Message']) > 60 else row['Message']
            
            # Tandai event penting
            marker = ""
            if "Ticket Has Been Reopened" in msg:
                marker = "🔄 "
            elif config.CUSTOMER_LEAVE_KEYWORD in msg:
                marker = "🚪 "
            elif "customer" in role.lower() and "?" in msg:
                marker = "❓ "
            elif "operator" in role.lower():
                marker = "👨‍💼 "
            
            print(f"      {time} {marker}[{role}]: {msg}")
        
        # Lanjutkan dengan analisis normal...
        is_complaint, complaint_data = self._is_complaint_ticket(ticket_id)
        if is_complaint:
            print("   🚨 COMPLAINT ticket detected")
            return self._analyze_complaint_replies(ticket_id, ticket_df, qa_pairs, main_issue, complaint_data)
        
        has_reopened, reopened_time = self._has_ticket_reopened_with_time(ticket_df)
        
        if has_reopened:
            is_claimed = self._has_claimed_after_reassigned(ticket_df, reopened_time)
            is_reassigned = self._has_reassigned_after_reopened(ticket_df, reopened_time)
            
            if is_reassigned:
                # Cek apakah masih ada percakapan setelah reassigned
                has_conversation_after_reassigned = self._has_conversation_after_reassigned(ticket_df, reopened_time)
                
                if has_conversation_after_reassigned:
                    print("   🔄 REOPENED, REASSIGNED, but HAS MORE CONVERSATIONS - treating as SERIOUS")
                    return self._analyze_enhanced_serious_replies(ticket_df, qa_pairs, main_issue)
                else:
                    print("   🔄 REOPENED and REASSIGNED - treating as NORMAL")
                    return self._analyze_normal_replies(ticket_df, qa_pairs, main_issue)
            
            if is_claimed:
                print("   🔄 REASSIGNED but CLAIMED - treating as SERIOUS (has continued conversation)")
                return self._analyze_enhanced_serious_replies(ticket_df, qa_pairs, main_issue)
            else:
                print("   ⚠️  SERIOUS ticket detected (has reopened pattern without reassigned)")
                
                serious_result = self._analyze_enhanced_serious_replies(ticket_df, qa_pairs, main_issue)
                
                # 🎯 MODIFIKASI PENTING: Cek customer leave untuk SERIOUS ticket
                if serious_result is None or serious_result.get('final_reply') is None:
                    print("   🔄 SERIOUS failed (no final reply) - checking for customer leave...")
                    
                    # Cek apakah ini customer leave
                    customer_leave_detected = self._check_enhanced_customer_leave_for_serious(
                        ticket_df, qa_pairs, main_issue, reopened_time
                    )
                    
                    if customer_leave_detected:
                        print("   🚨 SERIOUS + CUSTOMER LEAVE detected")
                        # Return special result untuk SERIOUS + CUSTOMER LEAVE
                        return self._create_serious_customer_leave_result(
                            ticket_df, qa_pairs, main_issue, reopened_time, serious_result
                        )
                    else:
                        print("   🔄 Not customer leave - falling back to NORMAL")
                        return self._analyze_normal_replies(ticket_df, qa_pairs, main_issue)
                
                return serious_result
        
        print("   ✅ NORMAL ticket detected")
        return self._analyze_normal_replies(ticket_df, qa_pairs, main_issue)
    def _create_serious_customer_leave_result(self, ticket_df, qa_pairs, main_issue, reopened_time, serious_result):
        """Create result untuk SERIOUS ticket yang berakhir dengan customer leave"""
        
        print("   📝 Creating SERIOUS + CUSTOMER LEAVE result...")
        
        # Cari first reply (jika ada dari serious_result)
        first_reply = None
        if serious_result and serious_result.get('first_reply'):
            first_reply = serious_result['first_reply']
        
        # Cari leave message setelah reopened
        leave_messages_after_reopen = ticket_df[
            (ticket_df['parsed_timestamp'] > reopened_time) &
            (ticket_df['Message'].str.contains(config.CUSTOMER_LEAVE_KEYWORD, na=False))
        ].sort_values('parsed_timestamp')
        
        if not leave_messages_after_reopen.empty:
            last_leave = leave_messages_after_reopen.iloc[-1]
            leave_time = last_leave['parsed_timestamp']
            leave_text = last_leave['Message']
            
            # Hitung lead time dari question_time ke leave
            lead_time_seconds = (leave_time - main_issue['question_time']).total_seconds()
            
            result = {
                'issue_type': 'serious_customer_leave',
                'first_reply': first_reply,
                'final_reply': {
                    'message': leave_text,
                    'timestamp': leave_time,
                    'lead_time_seconds': lead_time_seconds,
                    'lead_time_minutes': round(lead_time_seconds / 60, 2),
                    'lead_time_hhmmss': self._seconds_to_hhmmss(lead_time_seconds),
                    'note': 'Ticket ended with customer leave (no response after reopened)'
                },
                'customer_leave': True,
                'requirement_compliant': False,  # Karena tidak ada final reply yang proper
                'special_note': 'SERIOUS ticket reopened but customer did not respond'
            }
            
            print(f"   ✅ Created SERIOUS + CUSTOMER LEAVE result")
            print(f"   ⏰ Lead time to leave: {result['final_reply']['lead_time_minutes']} minutes")
            
            return result
        
        # Fallback jika tidak ada leave message (seharusnya tidak terjadi)
        return {
            'issue_type': 'serious_customer_leave',
            'first_reply': first_reply,
            'final_reply': None,
            'customer_leave': True,
            'requirement_compliant': False,
            'special_note': 'SERIOUS ticket - customer leave detected but no leave message'
        }
    
        def _check_enhanced_customer_leave_for_serious(self, ticket_df, qa_pairs, main_issue, reopened_time):
        """Cek customer leave khusus untuk SERIOUS tickets dengan reopened"""
        print(f"   🔍 Checking customer leave for SERIOUS ticket (reopened at: {reopened_time})")
        
        # 1. Cari leave messages SETELAH reopened
        leave_messages_after_reopen = ticket_df[
            (ticket_df['parsed_timestamp'] > reopened_time) &
            (ticket_df['Message'].str.contains(config.CUSTOMER_LEAVE_KEYWORD, na=False))
        ].sort_values('parsed_timestamp')
        
        if leave_messages_after_reopen.empty:
            print("   ⚠️ No leave messages found AFTER reopened")
            return False
        
        print(f"   📊 Found {len(leave_messages_after_reopen)} leave messages after reopened")
        
        # 2. Ambil leave message TERAKHIR setelah reopened
        last_leave_after_reopen = leave_messages_after_reopen.iloc[-1]
        leave_time = last_leave_after_reopen['parsed_timestamp']
        leave_text = last_leave_after_reopen['Message']
        
        print(f"   ⏰ Last leave after reopen: {leave_time}")
        print(f"   📝 Leave message: {leave_text[:80]}...")
        
        # 3. Cari operator messages SETELAH reopened dan SEBELUM leave
        operator_messages_after_reopen = ticket_df[
            (ticket_df['parsed_timestamp'] > reopened_time) &
            (ticket_df['parsed_timestamp'] < leave_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ].sort_values('parsed_timestamp')
        
        if operator_messages_after_reopen.empty:
            print("   ⚠️ No operator messages after reopen (ticket reopened but no operator response?)")
            
            # Cek apakah ada operator greeting SETELAH reopened
            operator_greetings_after_reopen = []
            for _, row in ticket_df.iterrows():
                if row['parsed_timestamp'] > reopened_time:
                    if self._is_operator_greeting(row['Message']):
                        operator_greetings_after_reopen.append(row)
            
            if operator_greetings_after_reopen:
                print(f"   👋 Found {len(operator_greetings_after_reopen)} operator greetings after reopen")
                last_op_greeting = operator_greetings_after_reopen[-1]
                last_op_time = last_op_greeting['parsed_timestamp']
            else:
                print("   ❌ No operator interaction after reopen at all")
                return False
        else:
            # Ambil operator message terakhir sebelum leave
            last_op_msg = operator_messages_after_reopen.iloc[-1]
            last_op_time = last_op_msg['parsed_timestamp']
            last_op_text = last_op_msg['Message'][:80] + '...'
            print(f"   👨‍💼 Last operator message after reopen: {last_op_time}")
            print(f"   💬 Message: {last_op_text}")
        
        # 4. Cek customer responses SETELAH operator terakhir (setelah reopened)
        customer_responses_after_last_op = ticket_df[
            (ticket_df['parsed_timestamp'] > last_op_time) &
            (ticket_df['parsed_timestamp'] < leave_time) &
            (ticket_df['Role'].str.lower().str.contains('customer', na=False))
        ]
        
        print(f"   👤 Customer responses after last operator (after reopen): {len(customer_responses_after_last_op)}")
        
        # Tampilkan customer responses jika ada
        if not customer_responses_after_last_op.empty:
            for i, (_, cust_row) in enumerate(customer_responses_after_last_op.iterrows()):
                print(f"      {i+1}. {cust_row['parsed_timestamp']}: {cust_row['Message'][:60]}...")
        
        # 5. 🎯 LOGIKA: Ini SERIOUS + CUSTOMER LEAVE jika:
        # - Ada leave message SETELAH reopened
        # - Customer TIDAK RESPON setelah operator terakhir (setelah reopened)
        # - Leave message menyebut "tidak ada respon"
        
        is_serious_customer_leave = (
            len(customer_responses_after_last_op) == 0 and
            "tidak ada respon" in leave_text.lower()
        )
        
        if is_serious_customer_leave:
            time_gap = (leave_time - last_op_time).total_seconds() / 60
            print(f"   🚨 CONFIRMED: SERIOUS + CUSTOMER LEAVE")
            print(f"   📝 Customer tidak respon {time_gap:.1f} menit setelah operator reply")
        else:
            if len(customer_responses_after_last_op) > 0:
                print(f"   ✅ NOT customer leave: Customer responded after reopened")
            elif "tidak ada respon" not in leave_text.lower():
                print(f"   ❓ Leave message doesn't mention 'no response'")
        
        return is_serious_customer_leave
    
    def _has_ticket_reopened_with_time(self, ticket_df):
        for _, row in ticket_df.iterrows():
            if "Ticket Has Been Reopened by" in str(row.get('Message', '')):
                reopened_time = row['parsed_timestamp']
                return True, reopened_time
        return False, None
    
    def _has_conversation_after_reassigned(self, ticket_df, reopened_time):
        """
        Cek apakah masih ada percakapan setelah reassigned.
        Percakapan didefinisikan sebagai message yang bukan system message.
        """
        if reopened_time is None:
            return False
        
        after_reopened = ticket_df[ticket_df['parsed_timestamp'] > reopened_time]
        
        if after_reopened.empty:
            return False
        
        # Cari timestamp reassigned terakhir
        last_reassigned_time = None
        for _, row in after_reopened.iterrows():
            msg = str(row['Message']).lower()
            if any(p in msg for p in ['reassigned to']):
                last_reassigned_time = row['parsed_timestamp']
        
        if last_reassigned_time is None:
            return False
        
        # Cek apakah ada message setelah reassigned terakhir
        after_last_reassigned = ticket_df[ticket_df['parsed_timestamp'] > last_reassigned_time]
        
        if after_last_reassigned.empty:
            return False
        
        # Filter untuk message yang bukan system message
        system_patterns = [
            'reassigned to',
            'claimed by',
            'ticket has been reopened',
            'ticket has been closed',
            'status changed to'
        ]
        
        for _, row in after_last_reassigned.iterrows():
            msg = str(row['Message']).lower()
            # Jika message tidak mengandung pattern system, berarti ada percakapan
            if not any(pattern in msg for pattern in system_patterns):
                return True
        
        return False
    
    def _has_reassigned_after_reopened(self, ticket_df, reopened_time):
        if reopened_time is None:
            return False
        
        after_reopened = ticket_df[ticket_df['parsed_timestamp'] > reopened_time]
        
        if after_reopened.empty:
            return False
        
        reassigned_count = 0
        for _, row in after_reopened.iterrows():
            msg = str(row['Message']).lower()
            if any(p in msg for p in ['reassigned to']):
                reassigned_count += 1
        
        return reassigned_count > 0

    def _has_claimed_after_reassigned(self, ticket_df, reopened_time):
        if reopened_time is None:
            return False
        
        after_reopened = ticket_df[ticket_df['parsed_timestamp'] > reopened_time]
        
        # PERBAIKAN: Gunakan .empty untuk cek DataFrame
        if after_reopened.empty:
            return False
            
        for _, row in after_reopened.iterrows():
            msg = str(row['Message']).lower()
            if any(p in msg for p in ['claimed by system to']):
                return True
        return False

    def _find_ticket_reopened_time(self, ticket_df):
        for _, row in ticket_df.iterrows():
            if "Ticket Has Been Reopened by" in str(row.get('Message', '')):
                return row['parsed_timestamp']
        return None

    def _analyze_enhanced_serious_replies(self, ticket_df, qa_pairs, main_issue):
        print("   🔍 Analyzing ENHANCED SERIOUS replies...")
        
        reopened_time = self._find_ticket_reopened_time(ticket_df)
        if reopened_time is None:
            return None
        
        first_reply = self._find_serious_first_reply(ticket_df, main_issue['question_time'], reopened_time)
        final_reply = self._find_enhanced_serious_final_reply(ticket_df, reopened_time, main_issue)
        
        first_reply_found = first_reply is not None
        final_reply_found = final_reply is not None
        
        customer_leave = self._check_enhanced_customer_leave(
            ticket_df, first_reply_found, final_reply_found, main_issue['question_time']
        )
        
        result = {
            'issue_type': 'serious',
            'first_reply': first_reply,
            'final_reply': final_reply,
            'customer_leave': customer_leave,
            'requirement_compliant': first_reply_found and final_reply_found
        }
        
        return result

    def _find_enhanced_serious_final_reply(self, ticket_df, reopened_time, main_issue):
        operator_messages = ticket_df[
            (ticket_df['parsed_timestamp'] > reopened_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ].sort_values('parsed_timestamp')

        # PERBAIKAN: Gunakan .empty untuk cek DataFrame
        if operator_messages.empty:
            return None

        solution_replies = []
        for _, msg in operator_messages.iterrows():
            if self._is_proper_solution_reply(msg['Message']):
                lt = (msg['parsed_timestamp'] - main_issue['question_time']).total_seconds()
                solution_replies.append({
                    'message': msg['Message'],
                    'timestamp': msg['parsed_timestamp'],
                    'lead_time_seconds': lt,
                    'lead_time_minutes': round(lt/60, 2),
                    'lead_time_hhmmss': self._seconds_to_hhmmss(lt),
                    'note': 'Proper solution reply after ticket reopened'
                })

        if solution_replies:
            return solution_replies[0]
        return None

    def _is_proper_solution_reply(self, message):
        if not message or len(message.strip()) < 10:
            return False
            
        msg = message.lower()
        
        # Skip questions
        if any(q in msg for q in ['?', 'apa ', 'bagaimana ', 'berapa ', 'kapan ', 'dimana ', 'kenapa ', 'mengapa ']):
            return False
        
        # Skip greetings
        if any(g in msg for g in ['selamat pagi', 'selamat siang', 'halo', 'hai']):
            return False
        
        # Minimal panjang
        if len(msg.split()) < 8:
            return False
            
        # PERBAIKAN: Gunakan extended keywords
        return any(k in msg for k in (self.solution_keywords_extended + config.ACTION_KEYWORDS))

    def _find_solution_reply(self, ticket_df, question_time):
        """Original method - tetap dipertahankan untuk compatibility"""
        operator_messages = ticket_df[
            (ticket_df['parsed_timestamp'] > question_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ].sort_values('parsed_timestamp')
        
        # PERBAIKAN: Gunakan .empty untuk cek DataFrame
        if operator_messages.empty:
            return None
            
        for _, msg in operator_messages.iterrows():
            if not any(c in msg['Message'].lower() for c in config.CLOSING_PATTERNS):
                if self._contains_solution_keyword_extended(msg['Message']):
                    lt = (msg['parsed_timestamp'] - question_time).total_seconds()
                    return {
                        'message': msg['Message'],
                        'timestamp': msg['parsed_timestamp'],
                        'lead_time_seconds': lt,
                        'lead_time_minutes': round(lt/60,2),
                        'lead_time_hhmmss': self._seconds_to_hhmmss(lt),
                        'note': 'Contains solution'
                    }
        return None

    def _find_serious_first_reply(self, ticket_df, question_time, reopened_time):
        print("   🔍 Finding serious FIRST reply...")
        operator_messages = ticket_df[
            (ticket_df['parsed_timestamp'] > question_time) &
            (ticket_df['parsed_timestamp'] < reopened_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ].sort_values('parsed_timestamp')
        
        # PERBAIKAN: Gunakan .empty untuk cek DataFrame
        if operator_messages.empty:
            return None
            
        for _, msg in operator_messages.iterrows():
            if self._contains_action_keyword(msg['Message']):
                lt = (msg['parsed_timestamp'] - question_time).total_seconds()
                return {
                    'message': msg['Message'],
                    'timestamp': msg['parsed_timestamp'],
                    'lead_time_seconds': lt,
                    'lead_time_minutes': round(lt/60,2),
                    'lead_time_hhmmss': self._seconds_to_hhmmss(lt)
                }
        return None

    def _contains_action_keyword(self, message):
        msg = str(message).lower()
        return any(k in msg for k in config.ACTION_KEYWORDS)
        
    def _contains_solution_keyword(self, message):
        """Original method - untuk compatibility"""
        msg = str(message).lower()
        return any(k in msg for k in config.SOLUTION_KEYWORDS)

    def _seconds_to_hhmmss(self, sec):
        if sec is None:
            return "00:00:00"
        try:
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        except:
            return "00:00:00"
                 
class CompleteAnalysisPipeline:
    def __init__(self, complaint_data_path=None):
        self.preprocessor = DataPreprocessor()
        self.parser = ConversationParser()
        self.issue_detector = MainIssueDetector()
        self.complaint_tickets = {}
        self.complaint_data_path = complaint_data_path
        self.failure_reasons = {}  # NEW: Track failure reasons
        
        print("🚀 ENHANCED Complete Analysis Pipeline Initialized")
    
    def analyze_single_ticket(self, ticket_df, ticket_id):
        """Analisis lengkap untuk single ticket - DENGAN BETTER ERROR HANDLING"""
        print(f"🎯 Analyzing Ticket: {ticket_id}")
        
        try:
            # Step 1: Parse Q-A pairs
            qa_pairs = self.parser.parse_conversation(ticket_df)
            
            if not qa_pairs:
                print("   ❌ No Q-A pairs detected")
                return self._create_ticket_result(ticket_id, "failed", "No meaningful Q-A pairs detected", {})
            
            print(f"   ✓ Found {len(qa_pairs)} Q-A pairs")
            
            # Step 2: Detect main issue
            main_issue = self.issue_detector.detect_main_issue(qa_pairs, ticket_df)
            
            if not main_issue:
                print("   ❌ No main issue detected")
                return self._create_ticket_result(ticket_id, "failed", "No suitable main issue detected", {})
            
            print(f"   ✓ Main issue detected (Score: {main_issue['detection_score']}): {main_issue['question'][:50]}...")
            
            # Step 3: Analyze replies
            if not hasattr(self, 'reply_analyzer'):
                self.reply_analyzer = ReplyAnalyzer(self.complaint_tickets)
                
            reply_analysis = self.reply_analyzer.analyze_replies(ticket_id, ticket_df, qa_pairs, main_issue)
            
            if not reply_analysis:
                print("   ❌ Reply analysis failed")
                return self._create_ticket_result(ticket_id, "failed", "Reply analysis failed", {})
            
            print(f"   ✓ Reply analysis: {reply_analysis['issue_type']}")
            
            # Step 4: Compile results
            result = self._compile_ticket_result(
                ticket_id, ticket_df, qa_pairs, main_issue, reply_analysis
            )
            
            print(f"   ✅ Analysis completed")
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"   ❌ Analysis failed: {error_msg}")
            return self._create_ticket_result(ticket_id, "failed", error_msg, {})
        
    def _compile_ticket_result(self, ticket_id, ticket_df, qa_pairs, main_issue, reply_analysis):
        """Compile ticket result dengan main issue yang enhanced"""
        # Hitung quality score
        quality_score = 0
        if reply_analysis['first_reply']:
            quality_score += 2
        if reply_analysis['final_reply']:
            quality_score += 2
        if not reply_analysis['customer_leave']:
            quality_score += 1
        if reply_analysis['requirement_compliant']:
            quality_score += 1
        
        # Tentukan performance rating
        if quality_score >= 5:
            performance_rating = 'excellent'
        elif quality_score >= 4:
            performance_rating = 'good'
        elif quality_score >= 2:
            performance_rating = 'fair'
        else:
            performance_rating = 'poor'
        
        # Ensure lead times are numeric
        first_lead_minutes = None
        final_lead_minutes = None
        
        if reply_analysis['first_reply'] and reply_analysis['first_reply'].get('lead_time_minutes'):
            try:
                first_lead_minutes = float(reply_analysis['first_reply']['lead_time_minutes'])
            except (ValueError, TypeError):
                first_lead_minutes = None
        
        if reply_analysis['final_reply'] and reply_analysis['final_reply'].get('lead_time_minutes'):
            try:
                final_lead_minutes = float(reply_analysis['final_reply']['lead_time_minutes'])
            except (ValueError, TypeError):
                final_lead_minutes = None
        
        result = {
            'ticket_id': ticket_id,
            'status': 'success',
            'analysis_timestamp': datetime.now(),
            
            # Conversation info
            'total_messages': len(ticket_df),
            'total_qa_pairs': len(qa_pairs),
            'answered_pairs': sum(1 for pair in qa_pairs if pair['is_answered']),
            'customer_leave': reply_analysis['customer_leave'],
            
            # Enhanced main issue info
            'main_question': main_issue['question'],
            'main_question_time': main_issue['question_time'],
            'main_issue_detection_score': main_issue['detection_score'],
            'main_issue_detection_reason': main_issue['detection_reason'],
            'final_issue_type': reply_analysis['issue_type'],
            
            # Reply analysis
            'first_reply_found': reply_analysis['first_reply'] is not None,
            'final_reply_found': reply_analysis['final_reply'] is not None,
            'first_reply_message': reply_analysis['first_reply']['message'] if reply_analysis['first_reply'] else None,
            'first_reply_time': reply_analysis['first_reply']['timestamp'] if reply_analysis['first_reply'] else None,
            'final_reply_message': reply_analysis['final_reply']['message'] if reply_analysis['final_reply'] else None,
            'final_reply_time': reply_analysis['final_reply']['timestamp'] if reply_analysis['final_reply'] else None,
            
            # Lead times
            'first_reply_lead_time_minutes': first_lead_minutes,
            'final_reply_lead_time_minutes': final_lead_minutes,
            'first_reply_lead_time_hhmmss': reply_analysis['first_reply'].get('lead_time_hhmmss') if reply_analysis['first_reply'] else None,
            'final_reply_lead_time_hhmmss': reply_analysis['final_reply'].get('lead_time_hhmmss') if reply_analysis['final_reply'] else None,
            
            # Untuk complaint
            'final_reply_lead_time_days': reply_analysis['final_reply'].get('lead_time_days') if reply_analysis['final_reply'] else None,
            
            # Performance metrics
            'performance_rating': performance_rating,
            'quality_score': quality_score,
            'quality_rating': 'good' if quality_score >= 4 else 'fair' if quality_score >= 2 else 'poor',
            'requirement_compliant': reply_analysis['requirement_compliant'],
            
            # Raw data
            '_raw_qa_pairs': qa_pairs,
            '_raw_reply_analysis': reply_analysis
        }
        
        return result
    
    def _create_ticket_result(self, ticket_id, status, reason, extra_data):
        """Create standardized result object"""
        result = {
            'ticket_id': ticket_id,
            'status': status,
            'failure_reason': reason if status == 'failed' else None,
            'analysis_timestamp': datetime.now()
        }
        result.update(extra_data)
        return result
    
    def analyze_all_tickets(self, df, max_tickets=None):
        """Analisis semua tickets dengan complaint matching yang benar"""
        print("🚀 STARTING COMPLETE ANALYSIS PIPELINE")
        
        # Load complaint data setelah punya raw_df
        if self.complaint_data_path and os.path.exists(self.complaint_data_path):
            complaint_df = self.preprocessor.load_complaint_data(self.complaint_data_path)
            self.complaint_tickets = self.preprocessor.match_complaint_tickets(df, complaint_df)
        
        # Initialize reply analyzer dengan complaint_tickets yang sudah di-load
        self.reply_analyzer = ReplyAnalyzer(self.complaint_tickets)
        
        ticket_ids = df['Ticket Number'].unique()
        
        if max_tickets:
            ticket_ids = ticket_ids[:max_tickets]
            print(f"🔍 Analyzing {max_tickets} tickets (max limit)...")
        else:
            print(f"🔍 Analyzing {len(ticket_ids)} tickets...")
        
        self.results = []
        successful_analyses = 0
        
        for i, ticket_id in enumerate(ticket_ids):
            ticket_df = df[df['Ticket Number'] == ticket_id]
            
            result = self.analyze_single_ticket(ticket_df, ticket_id)
            self.results.append(result)
            
            if result['status'] == 'success':
                successful_analyses += 1
            
            # Progress reporting
            if (i + 1) % 10 == 0 or (i + 1) == len(ticket_ids):
                progress = (i + 1) / len(ticket_ids) * 100
                print(f"   📊 Progress: {i + 1}/{len(ticket_ids)} ({progress:.1f}%) - {successful_analyses} successful")
        
        # Calculate statistics
        self.analysis_stats = self._calculate_stats(len(ticket_ids))
        
        print(f"\n🎉 ANALYSIS PIPELINE COMPLETED!")
        self._print_summary_report()
        
        return self.results, self.analysis_stats
    def _create_failed_result(self, ticket_id, reason):
        """Create failed result dengan detailed reason"""
        print(f"   ❌ TICKET FAILED: {reason}")
        
        # Track failure reason untuk reporting
        if reason not in self.failure_reasons:
            self.failure_reasons[reason] = []
        self.failure_reasons[reason].append(ticket_id)
        
        return {
            'ticket_id': ticket_id,
            'status': 'failed',
            'failure_reason': reason,
            'analysis_timestamp': datetime.now()
        }
        
    def _calculate_stats(self, total_tickets):
        """Hitung statistics dari results - FIXED VERSION dengan semua key yang diperlukan"""
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']
        
        # INITIALIZE SEMUA KEY YANG DIPERLUKAN
        stats = {
            'total_tickets': len(self.results),
            'successful_analysis': len(successful),
            'failed_analysis': len(failed),
            'success_rate': len(successful) / len(self.results) if self.results else 0,
            'failure_reasons': self.failure_reasons,
            
            # INITIALIZE EMPTY DICTIONARIES untuk menghindari KeyError
            'issue_type_distribution': {},
            'performance_distribution': {},
            'lead_time_stats': {
                'first_reply_avg_minutes': 0,
                'final_reply_avg_minutes': 0,
                'first_reply_samples': 0,
                'final_reply_samples': 0
            },
            'reply_effectiveness': {
                'first_reply_found_rate': 0,
                'final_reply_found_rate': 0,
                'customer_leave_cases': 0
            }
        }
        
        if successful:
            # Issue type distribution
            issue_types = [r['final_issue_type'] for r in successful if 'final_issue_type' in r]
            if issue_types:
                stats['issue_type_distribution'] = dict(Counter(issue_types))
            
            # Performance metrics
            performance_ratings = [r['performance_rating'] for r in successful if 'performance_rating' in r]
            if performance_ratings:
                stats['performance_distribution'] = dict(Counter(performance_ratings))
            
            # Lead time statistics
            first_lead_times = []
            final_lead_times = []
            
            for r in successful:
                # First reply lead times
                first_lt = r.get('first_reply_lead_time_minutes')
                if first_lt is not None and first_lt != 'N/A':
                    try:
                        first_lead_times.append(float(first_lt))
                    except (ValueError, TypeError):
                        pass
                
                # Final reply lead times (hanya untuk normal/serious, bukan complaint)
                final_lt = r.get('final_reply_lead_time_minutes')
                if final_lt is not None and final_lt != 'N/A':
                    try:
                        final_lead_times.append(float(final_lt))
                    except (ValueError, TypeError):
                        pass
            
            # Update lead time stats jika ada data
            if first_lead_times:
                stats['lead_time_stats']['first_reply_avg_minutes'] = np.mean(first_lead_times)
                stats['lead_time_stats']['first_reply_samples'] = len(first_lead_times)
            
            if final_lead_times:
                stats['lead_time_stats']['final_reply_avg_minutes'] = np.mean(final_lead_times)
                stats['lead_time_stats']['final_reply_samples'] = len(final_lead_times)
            
            # Reply effectiveness
            first_reply_found = sum(1 for r in successful if r.get('first_reply_found', False))
            final_reply_found = sum(1 for r in successful if r.get('final_reply_found', False))
            customer_leave_cases = sum(1 for r in successful if r.get('customer_leave', False))
            
            stats['reply_effectiveness'] = {
                'first_reply_found_rate': first_reply_found / len(successful) if successful else 0,
                'final_reply_found_rate': final_reply_found / len(successful) if successful else 0,
                'customer_leave_cases': customer_leave_cases
            }
        
        return stats
   
    def _print_summary_report(self):
        """Print summary report - FIXED VERSION"""
        stats = self.analysis_stats
        
        print(f"📊 ANALYSIS SUMMARY REPORT")
        print(f"   • Total Tickets: {stats['total_tickets']}")
        print(f"   • Successful Analysis: {stats['successful_analysis']} ({stats['success_rate']*100:.1f}%)")
        
        if 'issue_type_distribution' in stats:
            print(f"   • Issue Types: {stats['issue_type_distribution']}")
        
        if 'lead_time_stats' in stats:
            lt_stats = stats['lead_time_stats']
            # PERBAIKAN: Gunakan get() dengan default value
            first_reply_avg = lt_stats.get('first_reply_avg_minutes', 0)
            final_reply_avg = lt_stats.get('final_reply_avg_minutes', 0)
            
            print(f"   • Avg First Reply: {first_reply_avg:.1f} min")
            
            # PERBAIKAN: Handle case dimana final_reply_avg_minutes tidak ada atau 0
            if final_reply_avg > 0 and final_reply_avg != float('inf'):
                print(f"   • Avg Final Reply: {final_reply_avg:.1f} min")
            else:
                print(f"   • Avg Final Reply: Mixed (minutes/days)")
            
            print(f"   • First Reply Samples: {lt_stats.get('first_reply_samples', 0)}")
            print(f"   • Final Reply Samples: {lt_stats.get('final_reply_samples', 0)}")
        
        if 'reply_effectiveness' in stats:
            eff = stats['reply_effectiveness']
            print(f"   • First Reply Found: {eff.get('first_reply_found_rate', 0)*100:.1f}%")
            print(f"   • Final Reply Found: {eff.get('final_reply_found_rate', 0)*100:.1f}%")
            print(f"   • Customer Leave Cases: {eff.get('customer_leave_cases', 0)}")

# Results Exporter
class ResultsExporter:
    def __init__(self):
        self.output_dir = "output/"
        Path(self.output_dir).mkdir(exist_ok=True)
    
    def export_comprehensive_results(self, results, stats):
        """Export results ke Excel - PERBAIKAN: lebih lengkap"""
        try:
            output_path = f"{self.output_dir}analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet 1: Detailed Results
                detailed_data = []
                for result in results:
                    if result['status'] == 'success':
                        detailed_data.append({
                            'Ticket_ID': result['ticket_id'],
                            'Issue_Type': result['final_issue_type'],
                            'Main_Question': result['main_question'],
                            'Main_Question_Time': result.get('main_question_time'),
                            'First_Reply_Found': result['first_reply_found'],
                            'First_Reply_Message': result.get('first_reply_message', '')[:500] + '...' if result.get('first_reply_message') else 'Not found',
                            'First_Reply_Time': result.get('first_reply_time'),
                            'First_Reply_Lead_Time_Min': result.get('first_reply_lead_time_minutes'),
                            'First_Reply_Lead_Time_Format': result.get('first_reply_lead_time_hhmmss'),
                            'Final_Reply_Found': result['final_reply_found'],
                            'Final_Reply_Message': result.get('final_reply_message', '')[:500] + '...' if result.get('final_reply_message') else 'Not found',
                            'Final_Reply_Time': result.get('final_reply_time'),
                            'Final_Reply_Lead_Time_Min': result.get('final_reply_lead_time_minutes'),
                            'Final_Reply_Lead_Time_Days': result.get('final_reply_lead_time_days'),
                            'Final_Reply_Lead_Time_Format': result.get('final_reply_lead_time_hhmmss'),
                            'Performance_Rating': result['performance_rating'],
                            'Quality_Score': result['quality_score'],
                            'Quality_Rating': result['quality_rating'],
                            'Customer_Leave': result['customer_leave'],
                            'Requirement_Compliant': result['requirement_compliant'],
                            'Total_Messages': result['total_messages'],
                            'Total_QA_Pairs': result['total_qa_pairs'],
                            'Answered_Pairs': result['answered_pairs']
                        })
                    else:
                        detailed_data.append({
                            'Ticket_ID': result['ticket_id'],
                            'Issue_Type': 'FAILED',
                            'Main_Question': result.get('failure_reason', 'Analysis failed'),
                            'First_Reply_Found': False,
                            'Final_Reply_Found': False,
                            'Performance_Rating': 'FAILED',
                            'Quality_Score': 0
                        })
                
                if detailed_data:
                    df_detailed = pd.DataFrame(detailed_data)
                    df_detailed.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                # Sheet 2: Q-A Pairs Raw Data
                qa_pairs_data = []
                for result in results:
                    if result['status'] == 'success' and '_raw_qa_pairs' in result:
                        for i, qa_pair in enumerate(result['_raw_qa_pairs']):
                            qa_pairs_data.append({
                                'Ticket_ID': result['ticket_id'],
                                'QA_Pair_Index': i + 1,
                                'Question': qa_pair.get('question', ''),
                                'Question_Time': qa_pair.get('question_time'),
                                'Is_Answered': qa_pair.get('is_answered', False),
                                'Answer': qa_pair.get('answer', ''),
                                'Answer_Time': qa_pair.get('answer_time'),
                                'Lead_Time_Minutes': qa_pair.get('lead_time_minutes'),
                                'Lead_Time_Format': qa_pair.get('lead_time_hhmmss')
                            })
                
                if qa_pairs_data:
                    df_qa = pd.DataFrame(qa_pairs_data)
                    df_qa.to_excel(writer, sheet_name='Raw_QA_Pairs', index=False)
                
                # Sheet 3: Summary Statistics
                summary_data = self._create_summary_data(stats)
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False, header=False)
                
                # Sheet 4: Performance Metrics
                performance_data = self._create_performance_data(results)
                if performance_data:
                    df_perf = pd.DataFrame(performance_data)
                    df_perf.to_excel(writer, sheet_name='Performance_Metrics', index=False)
            
            print(f"💾 Results exported to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
            return None
    
    def _create_performance_data(self, results):
        """Create performance metrics data"""
        performance_data = []
        
        for result in results:
            if result['status'] == 'success':
                performance_data.append({
                    'Ticket_ID': result['ticket_id'],
                    'Issue_Type': result['final_issue_type'],
                    'Performance_Rating': result['performance_rating'],
                    'Quality_Score': result['quality_score'],
                    'Quality_Rating': result['quality_rating'],
                    'First_Reply_Found': result['first_reply_found'],
                    'Final_Reply_Found': result['final_reply_found'],
                    'Customer_Leave': result['customer_leave'],
                    'Requirement_Compliant': result['requirement_compliant'],
                    'Total_Messages': result['total_messages'],
                    'Answer_Rate': f"{(result['answered_pairs'] / result['total_qa_pairs']) * 100:.1f}%" if result['total_qa_pairs'] > 0 else '0%'
                })
        
        return performance_data
    
    def _create_summary_data(self, stats):
        """Create summary data untuk Excel"""
        summary_data = [
            ['ENHANCED ANALYSIS SUMMARY REPORT', ''],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['', ''],
            ['OVERALL STATISTICS', ''],
            ['Total Tickets', stats['total_tickets']],
            ['Successful Analysis', stats['successful_analysis']],
            ['Failed Analysis', stats['failed_analysis']],
            ['Success Rate', f"{stats['success_rate']*100:.1f}%"],
            ['', '']
        ]
        
        if 'issue_type_distribution' in stats:
            summary_data.append(['ISSUE TYPE DISTRIBUTION', ''])
            for issue_type, count in stats['issue_type_distribution'].items():
                percentage = (count / stats['successful_analysis']) * 100
                summary_data.append([f'{issue_type.title()} Issues', f'{count} ({percentage:.1f}%)'])
            summary_data.append(['', ''])
        
        if 'lead_time_stats' in stats:
            lt_stats = stats['lead_time_stats']
            summary_data.append(['LEAD TIME STATISTICS', ''])
            summary_data.append(['First Reply Avg (min)', f"{lt_stats['first_reply_avg_minutes']:.1f}"])
            summary_data.append(['Final Reply Avg (min)', f"{lt_stats['final_reply_avg_minutes']:.1f}"])
            summary_data.append(['First Reply Samples', lt_stats['first_reply_samples']])
            summary_data.append(['Final Reply Samples', lt_stats['final_reply_samples']])
            summary_data.append(['', ''])
        
        if 'reply_effectiveness' in stats:
            eff = stats['reply_effectiveness']
            summary_data.append(['REPLY EFFECTIVENESS', ''])
            summary_data.append(['First Reply Found Rate', f"{eff['first_reply_found_rate']*100:.1f}%"])
            summary_data.append(['Final Reply Found Rate', f"{eff['final_reply_found_rate']*100:.1f}%"])
            summary_data.append(['Customer Leave Cases', eff['customer_leave_cases']])
        
        return summary_data

# Initialize Pipeline
print("✅ ENHANCED Analysis Pipeline Ready!")
print("   ✓ New role handling (Ticket Automation & Blank)")
print("   ✓ New issue type detection logic")
print("   ✓ Complaint ticket matching")
print("   ✓ Ticket reopened detection")
print("=" * 60)











