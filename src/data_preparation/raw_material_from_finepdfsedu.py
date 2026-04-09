import re
from datasets import load_dataset
import json
import os

class QCHcuristicFilter:
    def __init__(self, threshold=5):
        self.threshold = threshold
        
        # High-weight: Almost certainly QC (Weight: 5)
        self.anchor_keywords = {
            r'qiskit', r'cirq', r'pennylane', r'openqasm', r'\\bra\{', r'\\ket\{', 
            r'\\braket', r'\\otimes', r'qubit', r'qutrit', r'bloch sphere', 
            r'shor\'s algorithm', r'grover\'s algorithm', r'surface code'
        }
        
        # Medium-weight: Highly likely QC (Weight: 2)
        self.technical_keywords = {
            r'hadamard', r'tofolli', r'entanglement', r'superposition', 
            r'decoherence', r'unitary operator', r'hamiltonian evolution', 
            r'quantum volume', r'nisq', r'error mitigation', r'vqe', r'qaoa'
        }
        
        # Low-weight: Common terms, need context (Weight: 1)
        self.contextual_keywords = {
            r'circuit', r'gate', r'measurement', r'fidelity', 
            r'noise', r'state', r'ancilla', r'simulation'
        }

    def calculate_score(self, text):
        score = 0
        text_lower = text.lower()
        
        # 1. Check Anchor Keywords (High Signal)
        for pattern in self.anchor_keywords:
            if re.search(pattern, text_lower):
                score += 5
        
        # 2. Check Technical Keywords (Medium Signal)
        for pattern in self.technical_keywords:
            if re.search(pattern, text_lower):
                score += 2
                
        # 3. Check Contextual Keywords (Low Signal)
        for pattern in self.contextual_keywords:
            if re.search(pattern, text_lower):
                score += 1
                
        return score

    def is_qc_related(self, text):
        return self.calculate_score(text) >= self.threshold

if __name__ == "__main__":
    
    if not os.path.exists("data/InstructionTuning/raw_materials"):
        os.makedirs("data/InstructionTuning/raw_materials")
   
    filter_tool = QCHcuristicFilter(threshold=20)

    dataset = load_dataset('HuggingFaceFW/finepdfs', split='train', streaming=True)

    counter = 0

    QC_corpus_chunk = []

    for sample in dataset:
        doc = sample['text']
        score = filter_tool.calculate_score(doc)
        passed = filter_tool.is_qc_related(doc)
        if passed:
            QC_corpus_chunk.append(sample)
            counter += 1
            if counter % 100 == 0:
                print(f"Processed {counter} QC-related documents so far...")
                # Save it to "data/InstructionTuning/raw_materials/QC_corpus_chunk_<counter>.jsonl"
                with open(f"data/InstructionTuning/raw_materials/QC_corpus_chunk_{counter}.jsonl", "w") as f:
                    for item in QC_corpus_chunk:
                        json.dump(item, f)
                        f.write("\n")
                # Clear the chunk for the next batch
                QC_corpus_chunk = []
        if counter >= 10000:  # Limit to 10000
            break

    print(f"Finished processing. Total QC-related documents collected: {counter}")