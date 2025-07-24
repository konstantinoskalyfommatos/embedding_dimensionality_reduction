import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from core.student import Student
from core.teacher import Teacher
from core.eval_functions import eval_intrinsic_original_vs_projected_space
from src.utils.custom_datasets import TokenizedDataset, EmbeddingsDataset
from utils.embedding_precalculation import get_precalculated_embeddings_dataset

sentences = ["Hello world", "Greetings", "How long is Limassol from here?", "And then the Cosmos was born."]

tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True)
encoder = SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True).to("cuda")
encoder.max_seq_length = 32

student_test_dataset = TokenizedDataset(
    torch.tensor(sentences),
    tokenizer=tokenizer,
    max_length=encoder.max_seq_length
)
student_test_loader = DataLoader(student_test_dataset, batch_size=256, shuffle=False)

# Teacher uses precomputed embeddings
with torch.no_grad():
    teacher_embeddings = encoder.encode(sentences, convert_to_tensor=True, device="cuda")
teacher_test_dataset = EmbeddingsDataset(teacher_embeddings)
teacher_test_loader = DataLoader(teacher_test_dataset, batch_size=256, shuffle=False)

# Projection net (sizes must match your trained model)
projection_net = torch.nn.Sequential(
    torch.nn.Linear(encoder.get_sentence_embedding_dimension(), 312),
    torch.nn.ReLU(),
    torch.nn.Linear(312, 256),
)

# Load student model
student = Student(
    backbone=encoder,
    projection_net=projection_net,
    freeze_backbone=True
)
student.load_state_dict(torch.load("storage/models/distilled_jina_v2_best.pth", map_location="cuda"))
student.eval()

# Teacher
teacher = Teacher(backbone=encoder, use_backbone=False)

# Evaluate intrinsic loss
loss = eval_intrinsic_original_vs_projected_space(
    student=student,
    student_val_loader=student_test_loader,
    teacher_val_loader=teacher_test_loader,
    device="cuda",
    positional_loss_factor=0.5,
    use_precalculated_student_embeddings=False
)
print(f"Intrinsic test Loss: {loss:.4f}")