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
        'apabila sudah cukup', 'apakah sudah cukup', 'apakah informasinya sudah cukup',
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

class MainIssueDetector:
    def __init__(self):
        # 1. Keywords Masalah
        self.problem_keywords = [
            'bagaimana', 'mobil saya', 'saya belum dapat', 'saya ingin menanyakan', 'stok', 'mobil saya mogok', 'apa bisa', 'masalah', 'problem', 'error', 'gagal', 'tidak bisa', 
            'kendala', 'gangguan', 'trouble', 'komplain', 'keluhan', 'kecewa',
            'rusak', 'blank', 'kosong', 'hang', 'lambat', 'eror', 'bug', 'bantu'
        ]
        
        # 2. Keywords Urgent
        self.urgent_keywords = [
            'mendesak', 'urgent', 'segera', 'penting', 'cepat', 'sekarang',
            'hari ini', 'besok', 'deadline', 'target'
        ]
        
        # 3. Keywords Komplain Keras
        self.complaint_keywords = [
            'komplain', 'kecewa', 'marah', 'protes', 'pengaduan', 'keluhan',
            'sakit hati', 'tidak puas', 'keberatan', 'sangat kecewa', 'anjir', 'bangsat'
        ]

        # 4. Pola Vague (Basa-basi)
        self.vague_patterns = [
            'mau nanya', 'mau tanya', 'boleh tanya', 'saya ingin bertanya', 
            'saya mau tanya', 'halo', 'hai', 'permisi', 'pagi', 'siang', 'sore', 'malam',
            'test', 'tes', 'ping', 'hallo', 'ok', 'oke', 'baik'
        ]

        # 5. [BARU] Blacklist Kata-kata Bot/Navigasi (Supaya tidak terpilih jadi main issue)
        self.bot_navigation_keywords = [
            'balik ke main menu', 'main menu', 'others', 'pusat bantuan', 
            'hubungi cs', 'setuju', 'tidak setuju', 'ya', 'tidak', 
            'live chat', 'chat dengan agen', 'menu utama', 'kembali'
        ]

    def _get_detection_reason(self, score):
        """Mapping score ke alasan"""
        reasons = {
            10: "Urgent problem detected",
            9: "Complaint detected", 
            8: "Problem keyword found",
            7: "Repeated question",
            6: "Detailed question",
            5: "Unanswered question",
            2: "Queue/Greeting detected (Contextual)",
            1: "Last interaction found (Fallback)",
            0: "Low confidence detection"
        }
        return reasons.get(score, "Unknown score")
    
    def detect_main_issue(self, qa_pairs, ticket_df):
        """
        Logika Utama:
        1. Cari titik mulai Operator (Start Time).
        2. Ambil chat SETELAH operator masuk.
        3. Jika kosong, ambil chat ANTRIAN (15 menit sebelum operator masuk).
        4. Jika operator ada tapi chat customer kosong/invalid -> Return None/Greeting (JANGAN ambil chat bot awal).
        """
        if not qa_pairs:
            print("   ❌ No Q-A pairs available")
            return None
        
        print(f"   🔍 Detecting main issue from {len(qa_pairs)} Q-A pairs...")
        
        # --- LANGKAH 1: Tentukan Waktu Mulai Operator ---
        conversation_start_time = self._find_conversation_start_time(ticket_df)
        
        target_qa_pairs = []
        source_type = "ALL" # Untuk debugging

        if conversation_start_time:
            print(f"   ⏰ Operator joined at: {conversation_start_time}")
            
            # A. Priority 1: Chat SETELAH Operator Masuk
            post_operator_pairs = [
                qa for qa in qa_pairs 
                if qa['question_time'] and qa['question_time'] >= conversation_start_time
            ]
            
            # Filter sampah bot navigation dari post_operator (jaga-jaga)
            post_operator_pairs = [
                qa for qa in post_operator_pairs 
                if not self._is_bot_navigation(qa['question'])
            ]
            
            if post_operator_pairs:
                print(f"   📝 Found {len(post_operator_pairs)} messages AFTER operator joined")
                target_qa_pairs = post_operator_pairs
                source_type = "POST_OP"
            else:
                # B. Priority 2: Chat Saat ANTRIAN (Queue Buffer 15 Menit)
                print("   ⚠️ No relevant message after operator. Checking queue buffer (15 mins before)...")
                queue_start_time = conversation_start_time - timedelta(minutes=15)
                
                queue_pairs = [
                    qa for qa in qa_pairs
                    if qa['question_time'] and queue_start_time <= qa['question_time'] < conversation_start_time
                ]
                
                # Filter sampah bot
                queue_pairs = [
                    qa for qa in queue_pairs 
                    if not self._is_bot_navigation(qa['question'])
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
            print("   ⚠️ No operator detected. Scanning all pairs excluding bot commands...")
            # Ambil semua tapi buang command bot
            target_qa_pairs = [
                qa for qa in qa_pairs 
                if not self._is_bot_navigation(qa['question'])
            ]
            source_type = "NO_OP"
        
        if not target_qa_pairs:
            return None

        # --- LANGKAH 2: Deteksi Masalah dari target_qa_pairs ---
        
        # Strategy 1: Strict (Ada keyword masalah jelas)
        main_issue = self._strict_main_issue_detection(target_qa_pairs)
        if main_issue:
            return main_issue
        
        # Strategy 2: Relaxed (Pertanyaan umum)
        print("   ⚠️ Strict detection failed, trying relaxed detection...")
        main_issue = self._relaxed_main_issue_detection(target_qa_pairs)
        if main_issue:
            return main_issue
        
        # Strategy 3: Contextual Fallback (Khusus jika sumbernya dari Queue/Post-Op)
        # Jika customer cuma bilang "Halo" di antrian, itu dianggap main issue (intent to connect)
        if source_type in ["POST_OP", "QUEUE"]:
            print(f"   ⚠️ Using contextual fallback from {source_type}...")
            # Ambil pesan terakhir customer yang bukan bot command
            last_qa = target_qa_pairs[-1]
            return {
                'question': last_qa['question'],
                'question_time': last_qa['question_time'],
                'detection_score': 2,
                'detection_reason': 'Customer greeting/queue message'
            }
        
        return None

    def _find_conversation_start_time(self, ticket_df):
        """Cari kapan percakapan benar-benar dimulai dengan operator"""
        ticket_df = ticket_df.sort_values('parsed_timestamp').reset_index(drop=True)
        
        # Regex yang mencakup log user ("Selamat malam... Selamat datang...")
        operator_greeting_patterns = [
            r"selamat\s+(pagi|siang|sore|malam).+selamat\s+datang", 
            r"selamat\s+datang\s+di\s+layanan",
            r"dengan\s+\w+\s*,?\s*apakah\s+ada",
            r"ada\s+yang\s+bisa\s+dibantu",
            r"perkenalkan.*nama\s+saya",
            r"saat\s+ini\s+anda\s+terhubung",
            r"terhubung\s+dengan\s+live\s+chat"
        ]
        
        for idx, row in ticket_df.iterrows():
            role = str(row['Role']).lower()
            message = str(row['Message'])
            
            # Cek apakah pengirim adalah manusia/agen
            is_agent = any(k in role for k in ['operator', 'agent', 'admin', 'cs', 'staff'])
            
            if is_agent:
                # Cek greeting pattern
                for pattern in operator_greeting_patterns:
                    if re.search(pattern, message, re.IGNORECASE):
                        print(f"   ✅ Found operator greeting: {message[:50]}...")
                        return row['parsed_timestamp']
                        
        # Fallback: Pesan pertama dari role Operator apapun
        for idx, row in ticket_df.iterrows():
            role = str(row['Role']).lower()
            if any(k in role for k in ['operator', 'agent', 'admin', 'cs']):
                print(f"   ⚠️ Using first operator message as anchor: {row['Message'][:30]}...")
                return row['parsed_timestamp']
                
        return None

    def _is_bot_navigation(self, text):
        """Cek apakah teks adalah command navigasi bot"""
        if not text: return True
        text_lower = text.lower().strip()
        
        # Cek exact match atau contain keyword bot
        for keyword in self.bot_navigation_keywords:
            if keyword == text_lower or (len(text_lower) < 20 and keyword in text_lower):
                return True
        return False

    def _calculate_question_score(self, question):
        score = 0
        question_lower = question.lower()
        words = question_lower.split()
        
        # Length score
        if len(words) >= 10: score += 2
        elif len(words) >= 5: score += 1
        
        # Keywords
        if any(k in question_lower for k in self.complaint_keywords): score += 5
        elif any(k in question_lower for k in self.urgent_keywords): score += 4
        elif any(k in question_lower for k in self.problem_keywords): score += 3
        
        # Indicators
        if '?' in question_lower: score += 1
        
        return score

    def _strict_main_issue_detection(self, qa_pairs):
        """Hanya return jika ada score tinggi"""
        candidates = []
        for qa in qa_pairs:
            # Skip yang terlalu pendek untuk strict
            if len(qa['question'].split()) < 3: continue
            
            score = self._calculate_question_score(qa['question'])
            if score >= 3: # Minimal ada problem keyword
                candidates.append((qa, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_qa, best_score = candidates[0]
            return {
                'question': best_qa['question'],
                'question_time': best_qa['question_time'],
                'detection_score': best_score,
                'detection_reason': self._get_detection_reason(best_score)
            }
        return None

    def _relaxed_main_issue_detection(self, qa_pairs):
        """Return pertanyaan terbaik yang masuk akal"""
        candidates = []
        for qa in qa_pairs:
            q_text = qa['question']
            
            # Skip 1 huruf/kata ga jelas
            if len(q_text.strip()) < 2: continue
            
            score = self._calculate_question_score(q_text)
            candidates.append((qa, score))
            
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_qa, best_score = candidates[0]
            
            # Update score jadi minimal 1 agar reason-nya bukan 'Unknown'
            final_score = max(best_score, 1)
            
            return {
                'question': best_qa['question'],
                'question_time': best_qa['question_time'],
                'detection_score': final_score,
                'detection_reason': self._get_detection_reason(final_score)
            }
        return None
    
class ReplyAnalyzer:
    def __init__(self, complaint_tickets=None):
        self.complaint_tickets = complaint_tickets or {}
        self.action_keywords = config.ACTION_KEYWORDS
        
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

    def _check_enhanced_customer_leave(self, ticket_df, first_reply_found, final_reply_found, question_time):
        """Enhanced customer leave detection dengan logika yang lebih akurat"""
        
        # Cari semua automation messages dengan keyword customer leave
        automation_messages = ticket_df[
            (ticket_df['Role'].str.lower().str.contains('automation', na=False)) &
            (ticket_df['Message'].str.contains(config.CUSTOMER_LEAVE_KEYWORD, na=False))
        ]
        
        # PERBAIKAN: Gunakan .empty untuk cek DataFrame
        if automation_messages.empty:
            return False
        
        # Ambil pesan automation pertama yang mengandung keyword leave
        leave_message = automation_messages.iloc[0]
        leave_time = leave_message['parsed_timestamp']
        
        print(f"   ⏰ Found automation leave message at: {leave_time}")
        
        # Cari operator greeting sebelum leave message
        operator_greetings = self._find_operator_greetings_before_time(ticket_df, leave_time)
        
        # PERBAIKAN: Gunakan .empty untuk cek DataFrame
        if operator_greetings.empty:
            print("   ⚠️ No operator greeting found before leave message")
            return False
        
        # Ambil greeting terakhir sebelum leave
        last_greeting = operator_greetings.iloc[-1]
        greeting_time = last_greeting['parsed_timestamp']
        
        print(f"   👋 Last operator greeting at: {greeting_time}")
        print(f"   ⏱️  Time gap: {(leave_time - greeting_time).total_seconds() / 60:.1f} minutes")
        
        # Cek apakah ada interaksi customer setelah greeting dan sebelum leave
        customer_interactions = self._find_customer_interactions_after_greeting(
            ticket_df, greeting_time, leave_time
        )
        
        # Cek apakah ada interaksi operator setelah greeting dan sebelum leave
        operator_interactions = self._find_operator_interactions_after_greeting(
            ticket_df, greeting_time, leave_time
        )
        
        print(f"   💬 Customer interactions after greeting: {len(customer_interactions)}")
        print(f"   👨‍💼 Operator interactions after greeting: {len(operator_interactions)}")
        
        # **LOGIKA UTAMA: Hanya dianggap customer leave jika:**
        # 1. Ada operator greeting
        # 2. TIDAK ADA interaksi customer setelah greeting
        # 3. TIDAK ADA interaksi operator meaningful setelah greeting (selain mungkin greeting ulang)
        # 4. Ada automation leave message
        
        is_true_leave = (
            len(customer_interactions) == 0 and 
            len(operator_interactions) == 0 and
            not operator_greetings.empty  # PERBAIKAN: Gunakan .empty
        )
        
        if is_true_leave:
            print("   ✅ TRUE CUSTOMER LEAVE: No interactions after operator greeting")
        else:
            print("   ❌ NOT customer leave: Found interactions after greeting")
            
        return is_true_leave

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
                'message': 'COMPLAINT_RESOLUTION',
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
        """Analyze replies untuk normal tickets"""
        final_reply = self._find_solution_reply(ticket_df, main_issue['question_time'])
        
        final_reply_found = final_reply is not None
        first_reply_found = False
        
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
        return analysis_result

    def analyze_replies(self, ticket_id, ticket_df, qa_pairs, main_issue):
        print(f"🔍 Analyzing replies for ticket {ticket_id}")
        
        is_complaint, complaint_data = self._is_complaint_ticket(ticket_id)
        if is_complaint:
            print("   🚨 COMPLAINT ticket detected")
            return self._analyze_complaint_replies(ticket_id, ticket_df, qa_pairs, main_issue, complaint_data)
        
        has_reopened, reopened_time = self._has_ticket_reopened_with_time(ticket_df)
        
        if has_reopened:
            is_reassigned = self._has_reassigned_after_reopened(ticket_df, reopened_time)
            if is_reassigned:
                print("   🔄 REOPENED but REASSIGNED - treating as NORMAL")
                return self._analyze_normal_replies(ticket_df, qa_pairs, main_issue)
            else:
                print("   ⚠️  SERIOUS ticket detected (has reopened pattern without reassigned)")
                
                serious_result = self._analyze_enhanced_serious_replies(ticket_df, qa_pairs, main_issue)
                
                # PERBAIKAN: Cek dengan benar untuk None
                if serious_result is None or serious_result.get('final_reply') is None:
                    print("   🔄 SERIOUS failed (no final reply) - falling back to NORMAL")
                    return self._analyze_normal_replies(ticket_df, qa_pairs, main_issue)
                
                return serious_result
        
        print("   ✅ NORMAL ticket detected")
        return self._analyze_normal_replies(ticket_df, qa_pairs, main_issue)

    def _has_ticket_reopened_with_time(self, ticket_df):
        for _, row in ticket_df.iterrows():
            if "Ticket Has Been Reopened by" in str(row.get('Message', '')):
                reopened_time = row['parsed_timestamp']
                return True, reopened_time
        return False, None

    def _has_reassigned_after_reopened(self, ticket_df, reopened_time):
        if reopened_time is None:
            return False
        
        after_reopened = ticket_df[ticket_df['parsed_timestamp'] > reopened_time]
        
        # PERBAIKAN: Gunakan .empty untuk cek DataFrame
        if after_reopened.empty:
            return False
            
        for _, row in after_reopened.iterrows():
            msg = str(row['Message']).lower()
            if any(p in msg for p in ['reassigned to', 'claimed by system to']):
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
        
        if any(q in msg for q in ['?', 'apa ', 'bagaimana ', 'berapa ', 'kapan ', 'dimana ', 'kenapa ', 'mengapa ']):
            return False
        
        if any(g in msg for g in ['selamat pagi', 'selamat siang', 'halo', 'hai']):
            return False
        
        if len(msg.split()) < 8:
            return False
            
        return any(k in msg for k in (config.SOLUTION_KEYWORDS + config.ACTION_KEYWORDS))

    def _find_solution_reply(self, ticket_df, question_time):
        operator_messages = ticket_df[
            (ticket_df['parsed_timestamp'] > question_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ].sort_values('parsed_timestamp')
        
        # PERBAIKAN: Gunakan .empty untuk cek DataFrame
        if operator_messages.empty:
            return None
            
        for _, msg in operator_messages.iterrows():
            if not any(c in msg['Message'].lower() for c in config.CLOSING_PATTERNS):
                if self._contains_solution_keyword(msg['Message']):
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


