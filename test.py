from essentia.standard import MonoLoader, TensorflowPredictMAEST

audioFile = input("Enter path to mp3 (320kps quality):")
try:
    audio = MonoLoader(filename=audioFile, sampleRate=32000, resampleQuality=4)()
except Exception as e:
    print(e)
else:
    model = TensorflowPredictMAEST(graphFilename="essentiaModel/discogs-maest-30s-pw-1.pb", output="StatefulPartitionedCall:7")
    embeddings = model(audio)
    print(embeddings)
