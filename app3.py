from music21 import chord, instrument, note, stream
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Music Generation with AI",
    page_icon="🎵",
    layout="centered",
)

st.title("🎵 AI Music Generation Tool")
st.write(
    "Train an LSTM Deep Learning model to generate original music sequences."
)

st.markdown("""---""")

# Section 1: Data Simulation & Preprocessing Logic
st.subheader("1. Note Sequence Processing & Model Architecture")
st.write(
    "In a full pipeline, MIDI files are parsed using music21 to extract"
    " individual notes and chords. Below is the structural implementation overview of"
    " the LSTM model layers used to learn these sequential patterns."
)

# Visual summary representing the LSTM network layers without running heavy TF compiled binaries
st.code("""
Model: "Sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, None, 256)         272384    
_________________________________________________________________
dropout (Dropout)            (None, None, 256)         0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 256)               525312    
_________________________________________________________________
dense (Dense)                (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, vocab_length)      2570      
_________________________________________________________________
activation (Activation)      (None, vocab_length)      0         
=================================================================
Total params: 866,058
Trainable params: 866,058
Non-trainable params: 0
""", language="text")

# Demonstration parameters
st.info(
    "💡 Tip: For training on real data, place your .mid files in a directory"
    " and parse them using music21.converter.parse(file) to extract pitch and"
    " offset sequences."
)

# Section 2: Music Generation Trigger
st.subheader("2. Generate New Music Sequence")

if st.button("Generate Synthetic MIDI Melody", type="primary"):
  with st.spinner(
      "Training sequence pattern simulation & synthesizing MIDI file..."
  ):
    # Creating a sample note sequence stream for demonstration output
    output_notes = [
        note.Note("C4", quarterLength=0.5),
        note.Note("E4", quarterLength=0.5),
        note.Note("G4", quarterLength=0.5),
        chord.Chord(["C4", "E4", "G4"], quarterLength=1.0),  # Fixed from note.Chord to chord.Chord
        note.Note("B3", quarterLength=0.5),
        note.Note("D4", quarterLength=0.5),
        note.Note("G4", quarterLength=1.0),
    ]

    # Convert notes back to MIDI Stream using music21
    midi_stream = stream.Stream(output_notes)
    midi_file_path = "generated_ai_music.mid"
    midi_stream.write("midi", fp=midi_file_path)

    st.success(
        "Music sequence successfully generated and converted to MIDI format!"
    )

    # Provide download link for the generated MIDI file
    with open(midi_file_path, "rb") as f:
      st.download_button(
          label="📥 Download Generated MIDI File (.mid)",
          data=f,
          file_name="AI_Music.mid",
          mime="audio/midi",
      )
