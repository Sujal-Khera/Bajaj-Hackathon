const { GridFSBucket, ObjectId } = require("mongodb");
const mongoose = require("mongoose");

const uploadToGridFS = async (req, res, next) => {
  try {
    const db = mongoose.connection.db;
    const bucket = new GridFSBucket(db, { bucketName: "uploads" });

    const { originalname, buffer } = req.file;
    const filename = req.body.filename || originalname;

    const uploadStream = bucket.openUploadStream(filename);
    uploadStream.end(buffer);

    uploadStream.on("finish", (file) => {
      req.fileId = file._id;
      req.storedName = file.filename;
      next();
    });

    uploadStream.on("error", (err) => {
      console.error("Upload error:", err);
      res.status(500).json({ message: "File upload error" });
    });
  } catch (err) {
    console.error("Exception in uploadToGridFS:", err);
    res.status(500).json({ message: "Internal Server Error" });
  }
};

const deleteFileById = async (req, res) => {
  const fileId = req.params.id;
  try {
    const db = mongoose.connection.db;
    const bucket = new GridFSBucket(db, { bucketName: "uploads" });
    await bucket.delete(new ObjectId(fileId));
    res.json({ message: "✅ File deleted", id: fileId });
  } catch (err) {
    console.error("❌ Error deleting file:", err);
    res.status(500).json({ message: "Deletion failed", error: err.message });
  }
};

const deleteAllFiles = async (req, res) => {
  try {
    const db = mongoose.connection.db;
    const bucket = new GridFSBucket(db, { bucketName: "uploads" });

    const files = await db.collection("uploads.files").find().toArray();
    const deleteOps = files.map(file => bucket.delete(file._id));
    await Promise.all(deleteOps);

    res.json({ message: `🗑️ Deleted ${files.length} files` });
  } catch (err) {
    console.error("❌ Error deleting all files:", err);
    res.status(500).json({ message: "Bulk deletion failed", error: err.message });
  }
};

module.exports = {
  uploadToGridFS,
  deleteFileById,
  deleteAllFiles,
};
