// Topic taxonomy for the graph. A video's topic is the description its recommendation text is most similar to
// (MiniLM cosine). Cheap and transparent, and good enough to browse by; it is not a claim of perfect tagging.
export const TOPICS = [
  ["pl", "Programming languages", "programming languages, compilers, type systems, functional programming, Lisp, Haskell, Rust, language design talks"],
  ["swe", "Software engineering", "software engineering, architecture, simplicity, testing, debugging, code quality, developer productivity talks"],
  ["systems", "Systems & performance", "operating systems, performance, CPUs, memory, low-level programming, kernels, latency, concurrency"],
  ["dist", "Databases & distributed systems", "databases, distributed systems, consensus, storage engines, SQL, scalability, cloud infrastructure"],
  ["security", "Security & privacy", "computer security, hacking, cryptography, exploits, privacy, surveillance, reverse engineering"],
  ["ai", "AI & machine learning", "artificial intelligence, machine learning, neural networks, deep learning, LLMs, robotics"],
  ["math", "Mathematics", "mathematics, proofs, geometry, number theory, statistics, visual math explanations"],
  ["physics", "Physics & astronomy", "physics, quantum mechanics, relativity, cosmology, astronomy, space science"],
  ["space", "Spaceflight", "rockets, spaceflight, NASA, SpaceX, satellites, Apollo, Mars missions"],
  ["bio", "Biology & medicine", "biology, medicine, health, neuroscience, genetics, evolution, the human body"],
  ["hardware", "Hardware & electronics", "electronics, hardware, chips, circuits, FPGA, retro computers, repair, engineering teardowns"],
  ["making", "Making & engineering", "machining, woodworking, manufacturing, how things are made, mechanical engineering, DIY builds"],
  ["history", "History & documentaries", "history, documentaries, historical events, computing history, old interviews and archival footage"],
  ["business", "Startups, business & economics", "startups, entrepreneurship, business, economics, finance markets, management, investing talks"],
  ["society", "Society & policy", "politics, policy, law, society, cities, urbanism, education, culture, media"],
  ["design", "Design & art", "design, user interfaces, typography, art, animation, film making, creativity"],
  ["games", "Games & game development", "video games, game development, game design, speedruns, demoscene, graphics programming"],
  ["music", "Music & audio", "music, musicians, audio engineering, synthesizers, music theory, concerts"],
  ["transport", "Aviation, cars & transport", "aviation, airplanes, cars, trains, ships, transportation engineering, accidents investigations"],
];
