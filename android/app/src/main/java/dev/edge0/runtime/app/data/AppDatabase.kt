// data/AppDatabase.kt
package dev.edge0.runtime.app.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

// v2: assistant token-id sidecar - messages.genTokens BLOB (additive ALTER).
private val MIGRATION_1_2 = object : Migration(1, 2) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE messages ADD COLUMN genTokens BLOB NULL")
    }
}

// v3: per-reply inline stats line - ttftMs/prefillTokS/memBytes columns.
private val MIGRATION_2_3 = object : Migration(2, 3) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE messages ADD COLUMN ttftMs INTEGER NULL")
        db.execSQL("ALTER TABLE messages ADD COLUMN prefillTokS REAL NULL")
        db.execSQL("ALTER TABLE messages ADD COLUMN memBytes INTEGER NULL")
    }
}

@Database(entities = [ThreadEntity::class, MessageEntity::class], version = 3,
          exportSchema = true)
abstract class AppDatabase : RoomDatabase() {
    abstract fun threads(): ThreadDao
    abstract fun messages(): MessageDao

    companion object {
        fun create(context: Context): AppDatabase =
            Room.databaseBuilder(context, AppDatabase::class.java, "edge0-chat.db")
                .addMigrations(MIGRATION_1_2, MIGRATION_2_3).build()
    }
}
