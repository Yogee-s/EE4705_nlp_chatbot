#!/usr/bin/env python3
"""
Enhanced Dataset Augmentation Script
Generates high-quality synthetic question-answer pairs using multiple techniques
"""

import pandas as pd
import random
import re
import argparse
from collections import Counter
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download("omw-1.4")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

class DatasetAugmenter:
    def __init__(self, preserve_entities=True):
        self.stop_words = set(stopwords.words('english'))
        self.preserve_entities = preserve_entities
        
        # Common university/academic entities to preserve
        self.entities_to_preserve = {
            'nus', 'yale-nus', 'usp', 'college', 'university', 'singapore',
            'engineering', 'computing', 'business', 'medicine', 'science',
            'faculty', 'school', 'department', 'course', 'module', 'semester',
            'year', 'student', 'undergraduate', 'graduate', 'phd', 'masters',
            'bachelor', 'degree', 'diploma', 'certificate', 'gpa', 'cap',
            'credits', 'units', 'prerequisite', 'corequisite', 'elective',
            'compulsory', 'mandatory', 'optional', 'requirements', 'admission',
            'application', 'fees', 'tuition', 'scholarship', 'bursary',
            'hostel', 'residence', 'campus', 'library', 'examination', 'exam'
        }
        
        # Question starters for reformulation
        self.question_starters = [
            "What", "How", "When", "Where", "Why", "Who", "Which", "Can", "Do", "Does",
            "Is", "Are", "Will", "Would", "Could", "Should", "Tell me about",
            "Explain", "Describe", "What are", "How do", "How can"
        ]

    def is_entity(self, word):
        """Check if a word should be preserved (entities, proper nouns, etc.)"""
        word_lower = word.lower()
        return (
            word_lower in self.entities_to_preserve or
            word.isupper() or  # Acronyms
            word[0].isupper() or  # Proper nouns
            word.isdigit() or  # Numbers
            any(char.isdigit() for char in word) or  # Contains digits
            len(word) <= 2  # Very short words
        )

    def get_synonyms(self, word, pos=None):
        """Get high-quality synonyms for a word"""
        if self.is_entity(word):
            return []
            
        synonyms = set()
        synsets = wordnet.synsets(word, pos=pos)
        
        for synset in synsets[:3]:  # Limit to top 3 synsets for quality
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                # Filter out low-quality synonyms
                if (synonym.lower() != word.lower() and
                    not synonym.lower() in self.stop_words and
                    len(synonym) > 2 and
                    synonym.isalpha()):
                    synonyms.add(synonym)
        
        return list(synonyms)

    def synonym_replacement(self, sentence, n=1):
        """Replace up to n words with high-quality synonyms"""
        words = word_tokenize(sentence)
        new_words = words.copy()
        
        # Filter words suitable for replacement
        replaceable_words = []
        for i, word in enumerate(words):
            if (len(word) > 3 and 
                word.lower() not in self.stop_words and
                word not in string.punctuation and
                not self.is_entity(word)):
                replaceable_words.append((i, word))
        
        if not replaceable_words:
            return sentence
            
        random.shuffle(replaceable_words)
        replacements_made = 0
        
        for idx, word in replaceable_words:
            if replacements_made >= n:
                break
                
            synonyms = self.get_synonyms(word)
            if synonyms:
                # Choose synonym that maintains similar length/complexity
                best_synonym = min(synonyms, key=lambda x: abs(len(x) - len(word)))
                new_words[idx] = best_synonym
                replacements_made += 1
        
        return ' '.join(new_words)

    def question_reformulation(self, question):
        """Reformulate questions using different structures"""
        question = question.strip()
        
        # Remove question mark for processing
        has_question_mark = question.endswith('?')
        if has_question_mark:
            question = question[:-1]
        
        question_lower = question.lower()
        
        # Pattern-based reformulations
        reformulations = []
        
        # "What is X" -> "Can you tell me about X", "Explain X"
        if question_lower.startswith('what is ') or question_lower.startswith('what are '):
            topic = question[8:] if question_lower.startswith('what is ') else question[9:]
            reformulations.extend([
                f"Can you tell me about {topic}",
                f"Explain {topic}",
                f"Tell me about {topic}",
                f"I'd like to know about {topic}"
            ])
        
        # "How do I" -> "What's the way to", "How can I"
        elif question_lower.startswith('how do i '):
            action = question[9:]
            reformulations.extend([
                f"What's the way to {action}",
                f"How can I {action}",
                f"What are the steps to {action}"
            ])
        
        # "Where can I" -> "What's the location of", "How do I find"
        elif question_lower.startswith('where can i '):
            action = question[12:]
            reformulations.extend([
                f"How do I find where to {action}",
                f"What's the location to {action}"
            ])
        
        # "Do I need" -> "Is it necessary to", "Must I"
        elif question_lower.startswith('do i need '):
            requirement = question[10:]
            reformulations.extend([
                f"Is it necessary to have {requirement}",
                f"Must I have {requirement}",
                f"Is {requirement} required"
            ])
        
        # Add question mark back
        reformulations = [q + '?' if not q.endswith('?') else q for q in reformulations]
        
        return reformulations if reformulations else [question + ('?' if has_question_mark else '')]

    def back_translation_simulation(self, sentence):
        """Simulate back-translation effects by making subtle changes"""
        words = sentence.split()
        new_words = []
        
        for word in words:
            # Simulate common back-translation patterns
            if random.random() < 0.1:  # 10% chance
                if word.lower() == 'can':
                    new_words.append(random.choice(['can', 'could', 'am able to']))
                elif word.lower() == 'will':
                    new_words.append(random.choice(['will', 'would', 'shall']))
                elif word.lower() == 'need':
                    new_words.append(random.choice(['need', 'require', 'must have']))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)

    def add_context_variations(self, question):
        """Add contextual variations to questions"""
        variations = []
        
        # Add politeness markers
        polite_starters = ["Could you please tell me", "I would like to know", 
                          "Can you help me understand", "I'm wondering"]
        if not any(question.lower().startswith(starter.lower()) for starter in polite_starters):
            for starter in random.sample(polite_starters, 2):
                variations.append(f"{starter} {question.lower()}")
        
        # Add urgency/context markers
        context_additions = ["for my application", "as a prospective student", 
                           "for planning purposes", "before I apply"]
        for addition in random.sample(context_additions, 1):
            base_q = question[:-1] if question.endswith('?') else question
            variations.append(f"{base_q} {addition}?")
        
        return variations

    def augment_dataset(self, df, multiplier=3, techniques=['synonym', 'reformulate', 'context']):
        """
        Augment dataset using multiple techniques
        
        Args:
            df: Input dataframe with 'question' and 'answer' columns
            multiplier: How many new samples to generate per original
            techniques: List of augmentation techniques to use
        """
        augmented_rows = []
        
        print(f"Starting augmentation with {len(df)} original samples...")
        print(f"Techniques: {techniques}")
        
        for i, row in df.iterrows():
            if pd.isna(row['question']) or pd.isna(row['answer']):
                continue
                
            original_q = str(row['question']).strip()
            original_a = str(row['answer']).strip()
            
            if not original_q or not original_a:
                continue
            
            # Always keep original
            augmented_rows.append({
                'question': original_q,
                'answer': original_a,
                'augmentation_type': 'original'
            })
            
            generated_questions = set()  # Avoid duplicates
            generated_questions.add(original_q.lower())
            
            attempts = 0
            successful_augmentations = 0
            
            while successful_augmentations < multiplier and attempts < multiplier * 3:
                attempts += 1
                
                # Choose random technique
                technique = random.choice(techniques)
                new_question = original_q
                
                try:
                    if technique == 'synonym':
                        new_question = self.synonym_replacement(original_q, n=random.randint(1, 2))
                        aug_type = 'synonym_replacement'
                    
                    elif technique == 'reformulate':
                        reformulations = self.question_reformulation(original_q)
                        if reformulations and len(reformulations) > 1:
                            new_question = random.choice(reformulations[1:])  # Skip first (original)
                        aug_type = 'reformulation'
                    
                    elif technique == 'context':
                        context_vars = self.add_context_variations(original_q)
                        if context_vars:
                            new_question = random.choice(context_vars)
                        aug_type = 'context_variation'
                    
                    elif technique == 'backtranslation':
                        new_question = self.back_translation_simulation(original_q)
                        aug_type = 'back_translation'
                    
                    # Quality check
                    if (new_question.lower() not in generated_questions and
                        len(new_question.strip()) > 5 and
                        new_question != original_q):
                        
                        augmented_rows.append({
                            'question': new_question.strip(),
                            'answer': original_a,
                            'augmentation_type': aug_type
                        })
                        generated_questions.add(new_question.lower())
                        successful_augmentations += 1
                        
                except Exception as e:
                    print(f"Error in augmentation: {e}")
                    continue
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(df)} samples...")
        
        return pd.DataFrame(augmented_rows)

def main():
    parser = argparse.ArgumentParser(description="Augment Q&A dataset")
    parser.add_argument("--input", default="data/nus_qna.csv", 
                       help="Input CSV file")
    parser.add_argument("--output", default="data/nus_qna_augmented.csv",
                       help="Output CSV file")
    parser.add_argument("--multiplier", type=int, default=3,
                       help="Number of augmented samples per original")
    parser.add_argument("--techniques", nargs="+", 
                       default=['synonym', 'reformulate', 'context'],
                       choices=['synonym', 'reformulate', 'context', 'backtranslation'],
                       help="Augmentation techniques to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("Enhanced Dataset Augmentation Script")
    print("=" * 50)
    
    # Load dataset
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} samples from {args.input}")
    except FileNotFoundError:
        print(f"Error: Could not find {args.input}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Validate required columns
    if 'question' not in df.columns or 'answer' not in df.columns:
        print("Error: Dataset must have 'question' and 'answer' columns")
        return
    
    # Initialize augmenter
    augmenter = DatasetAugmenter()
    
    # Perform augmentation
    print(f"\nStarting augmentation...")
    augmented_df = augmenter.augment_dataset(df, args.multiplier, args.techniques)
    
    # Save results
    augmented_df.to_csv(args.output, index=False)
    
    # Show statistics
    print(f"\nAugmentation Complete!")
    print(f"Original samples: {len(df)}")
    print(f"Augmented samples: {len(augmented_df)}")
    print(f"Total increase: {len(augmented_df) - len(df)} samples")
    print(f"Multiplication factor: {len(augmented_df) / len(df):.2f}x")
    
    # Show technique breakdown
    if 'augmentation_type' in augmented_df.columns:
        print("\nAugmentation technique breakdown:")
        technique_counts = augmented_df['augmentation_type'].value_counts()
        for technique, count in technique_counts.items():
            percentage = (count / len(augmented_df)) * 100
            print(f"  {technique}: {count} ({percentage:.1f}%)")
    
    print(f"\nAugmented dataset saved to: {args.output}")
    
    # Show some examples
    print("\nExample augmented questions:")
    non_original = augmented_df[augmented_df['augmentation_type'] != 'original'].sample(min(5, len(augmented_df)))
    for _, row in non_original.iterrows():
        print(f"  [{row['augmentation_type']}] {row['question']}")

if __name__ == "__main__":
    main()