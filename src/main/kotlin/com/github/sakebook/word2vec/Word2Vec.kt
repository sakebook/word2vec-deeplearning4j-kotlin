package com.github.sakebook.word2vec

import java.io.File
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Word not found. At least one more word.")
        return
    }
    word2vecWithD4J(args)
}

fun word2vecWithD4J(args: Array<String>) {
    val wordVectors = WordVectorSerializer.readWord2VecModel(File(ClassLoader.getSystemResource("ja.vec").file))
    val results = args.map { getWordVectorMatrix(wordVectors, it) }
    todo(results)
}

private fun getWordVectorMatrix(word2Vec: Word2Vec, word: String): Pair<String, INDArray?> {
    println("Target word is \"$word\"")
    if (!word2Vec.hasWord(word)) {
        println("$word is not vocab")
        return word to null
    }
    val result: INDArray? = word2Vec.getWordVectorMatrix(word)
    if (result == null) {
        println("$word is not table")
        return word to null
    }
    return word to result
}

fun todo(results: List<Pair<String, INDArray?>>) {
    results.forEach {
        println("${it.first}, ${it.second}")
    }
}

fun copyINDArray(indArray: INDArray): INDArray {
    val japanArray = doubleArrayOf(*indArray.data().asDouble())
    return Nd4j.create(japanArray)
}