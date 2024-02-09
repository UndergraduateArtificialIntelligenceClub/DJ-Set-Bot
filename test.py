from essentia.standard import MonoLoader, TensorflowPredictMAEST

audio = MonoLoader(filename="18025392_365_(Original Mix).mp3", sampleRate=32000, resampleQuality=4)()
model = TensorflowPredictMAEST(graphFilename="essentiaModel/discogs-maest-30s-pw-1.pb", output="StatefulPartitionedCall:7")
embeddings = model(audio)
print(embeddings)
