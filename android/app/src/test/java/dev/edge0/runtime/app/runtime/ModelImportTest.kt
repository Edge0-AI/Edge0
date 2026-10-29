// ModelImportTest - file-name gate and the partial-then-rename copy.
package dev.edge0.runtime.app.runtime

import java.io.ByteArrayInputStream
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelImportTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun fileNameKeepsAFlatGgufBasename() {
        assertEquals("edge0-8b.gguf", ModelImport.fileName("edge0-8b.gguf"))
        assertEquals("lora_edge0_8b-gguf.gguf",
            ModelImport.fileName("/storage/Download/lora_edge0_8b-gguf.gguf"))
        assertEquals("EDGE0-35B.gguf", ModelImport.fileName("EDGE0-35B.GGUF"))
    }

    @Test fun fileNameRejectsAnythingElse() {
        assertNull(ModelImport.fileName(null))
        assertNull(ModelImport.fileName("   "))
        assertNull(ModelImport.fileName("notes.txt"))
        assertNull(ModelImport.fileName(".gguf"))
        assertNull(ModelImport.fileName(".."))
        assertNull(ModelImport.fileName("a/../b"))
    }

    @Test fun copyRenamesOnlyAfterTheFullStream() = runBlocking {
        val dir = tmp.newFolder("models")
        val dest = ModelImport.copy(
            ByteArrayInputStream("gguf-bytes".toByteArray()),
            dir, "edge0-8b.gguf", sizeBytes = 10,
        )
        assertEquals("gguf-bytes", dest.readText())
        assertEquals("edge0-8b.gguf", dest.name)
        assertFalse(File(dir, "edge0-8b.gguf.partial").exists())
        val names = ModelInventory.scan(dir).map { it.name }
        assertEquals(listOf("edge0-8b"), names)
    }

    @Test fun truncatedStreamDeletesThePartial() = runBlocking {
        val dir = tmp.newFolder("models")
        val error = runCatching {
            ModelImport.copy(
                ByteArrayInputStream("short".toByteArray()),
                dir, "edge0-8b.gguf", sizeBytes = 10,
            )
        }.exceptionOrNull()
        assertTrue(error is ModelImport.ImportException)
        assertTrue(dir.listFiles().isNullOrEmpty())
    }

    @Test fun refusesWhenTheVolumeCannotHoldTheFile() = runBlocking {
        val dir = tmp.newFolder("models")
        val error = runCatching {
            ModelImport.copy(
                ByteArrayInputStream("x".toByteArray()),
                dir, "edge0-35b.gguf", sizeBytes = 100,
                availableBytes = 10,
            )
        }.exceptionOrNull()
        assertTrue(error is ModelImport.ImportException)
        assertTrue(dir.listFiles().isNullOrEmpty())
    }

    @Test fun stalePartialIsRemovedAndAFinishedFileIsKept() {
        val dir = tmp.newFolder("models")
        File(dir, "edge0-8b.gguf.partial").writeText("half")
        File(dir, "edge0-8b.gguf").writeText("whole")
        ModelImport.deleteStalePartials(dir)
        assertFalse(File(dir, "edge0-8b.gguf.partial").exists())
        assertEquals("whole", File(dir, "edge0-8b.gguf").readText())
    }

    @Test fun loraCompanionIsStoredButNotListed() = runBlocking {
        val dir = tmp.newFolder("models")
        ModelImport.copy(ByteArrayInputStream("lora".toByteArray()), dir, "lora_edge0_8b-gguf.gguf", 4)
        assertTrue(File(dir, "lora_edge0_8b-gguf.gguf").exists())
        assertTrue(ModelInventory.scan(dir).isEmpty())
    }
}
