import pandas as pd
import time
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

class FlipkartDescriptionGenerator:
    def __init__(self, input_csv_path, output_csv_path="enhanced_flipkart_products.csv"):
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        
        # Load existing processed data if available
        self.processed_df = self.load_processed_data()
        
        # Initialize LLM
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.load_qwen_model()
    
    def load_qwen_model(self):
        """Load Qwen2.5 model for text generation"""
        print("Loading Qwen2.5 model...")
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("✅ Qwen2.5 model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_processed_data(self):
        """Load existing processed data to avoid reprocessing"""
        if os.path.exists(self.output_csv_path):
            try:
                df = pd.read_csv(self.output_csv_path)
                print(f"📋 Loaded {len(df)} already processed items")
                return df
            except Exception as e:
                print(f"Warning: Could not load existing data: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    
    def is_already_processed(self, pid):
        """Check if product ID is already processed"""
        if len(self.processed_df) == 0:
            return False
        return pid in self.processed_df['pid'].values if 'pid' in self.processed_df.columns else False
    
    def clean_text(self, text):
        """Clean and format text data"""
        if pd.isna(text) or text == '' or text == 'No description available':
            return None
        
        # Convert to string and clean
        text = str(text)
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        return text if len(text) > 5 else None
    
    def extract_specifications(self, specs_text):
        """Extract key specifications from the specifications text"""
        if not specs_text or pd.isna(specs_text):
            return []
        
        try:
            # Try to parse as JSON first
            if specs_text.startswith('{') or specs_text.startswith('['):
                specs_data = json.loads(specs_text)
                if isinstance(specs_data, dict):
                    specs = []
                    for key, value in specs_data.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                specs.append(f"{sub_key}: {sub_value}")
                        else:
                            specs.append(f"{key}: {value}")
                    return specs[:5]  # Top 5 specs
            
            # If not JSON, try to extract key-value pairs
            specs = []
            lines = str(specs_text).split('\n')
            for line in lines:
                if ':' in line:
                    specs.append(line.strip())
                if len(specs) >= 5:
                    break
            
            return specs
            
        except:
            # Fallback: return first few lines
            return str(specs_text).split('\n')[:3]
    
    def generate_enhanced_description(self, row):
        """Generate enhanced description using product data"""
        try:
            # Extract product information
            product_name = self.clean_text(row.get('product_name', ''))
            category = self.clean_text(row.get('product_category_tree', ''))
            brand = self.clean_text(row.get('brand', ''))
            retail_price = row.get('retail_price', '')
            discounted_price = row.get('discounted_price', '')
            rating = row.get('product_rating', '')
            overall_rating = row.get('overall_rating', '')
            existing_desc = self.clean_text(row.get('description', ''))
            specifications = self.extract_specifications(row.get('product_specifications', ''))
            is_fk_advantage = row.get('is_FK_Advantage_product', False)
            product_url = row.get('product_url', '')
            
            if not product_name:
                return "Product name not available", "Description could not be generated", product_url
            
            # Build context for the model
            context_parts = []
            
            if brand:
                context_parts.append(f"Brand: {brand}")
            
            if category:
                # Clean category tree
                category_clean = category.replace(' >> ', ' > ').replace('["', '').replace('"]', '')
                context_parts.append(f"Category: {category_clean}")
            
            if retail_price and discounted_price:
                try:
                    retail = float(retail_price)
                    discounted = float(discounted_price)
                    if retail > discounted:
                        discount_pct = int(((retail - discounted) / retail) * 100)
                        context_parts.append(f"Price: ₹{discounted} (₹{retail}, {discount_pct}% off)")
                    else:
                        context_parts.append(f"Price: ₹{discounted}")
                except:
                    context_parts.append(f"Price: ₹{discounted_price}")
            elif discounted_price:
                context_parts.append(f"Price: ₹{discounted_price}")
            
            if rating:
                context_parts.append(f"Rating: {rating}")
            
            if specifications:
                specs_text = ". ".join(specifications[:3])  # Top 3 specs
                context_parts.append(f"Key Features: {specs_text}")
            
            if is_fk_advantage:
                context_parts.append("Flipkart Advantage Product")
            
            # Create the prompt - Fixed format for cleaner output
            context = ". ".join(context_parts)
            
            # Simplified prompt that should generate cleaner responses
            prompt = f"""Create a compelling product description in 2-3 sentences:

Product: {product_name}
Details: {context}

Description:"""
            
            # Generate description
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean the response - FIXED: Better extraction
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the description part after "Description:"
            if "Description:" in generated_text:
                description = generated_text.split("Description:")[-1].strip()
            else:
                # Fallback: remove the original prompt
                description = generated_text.replace(prompt, "").strip()
            
            # Clean up the description
            description = re.sub(r'\n+', ' ', description)
            description = re.sub(r'\s+', ' ', description)
            description = description.strip()
            
            # Remove any remaining prompt artifacts
            description = re.sub(r'^(Product:|Details:|Create|Write|Generate).*?:', '', description, flags=re.IGNORECASE)
            description = description.strip()
            
            # Take only the first few sentences and ensure it ends properly
            sentences = [s.strip() for s in description.split('.') if s.strip()]
            if len(sentences) > 3:
                description = '. '.join(sentences[:3]) + '.'
            elif sentences:
                description = '. '.join(sentences)
                if not description.endswith('.'):
                    description += '.'
            
            # Final cleanup and fallback
            if len(description) < 20 or not description:
                if brand:
                    description = f"High-quality {product_name} from {brand}. Perfect for your needs with excellent features and reliable performance."
                else:
                    description = f"Premium {product_name} offering excellent quality and features. Great value for money with reliable performance."
            
            return product_name, description, product_url
            
        except Exception as e:
            print(f"❌ Error generating description: {e}")
            return row.get('product_name', 'Unknown Product'), "Description generation failed", row.get('product_url', '')
    
    def process_products(self, max_items=None):
        """Process products from the input CSV"""
        print(f"📂 Loading input CSV: {self.input_csv_path}")
        
        try:
            # Load input CSV
            input_df = pd.read_csv(self.input_csv_path)
            print(f"📊 Found {len(input_df)} products in input file")
            
            # Check required columns
            required_cols = ['product_name', 'pid']
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                return
            
            print("✅ All required columns found")
            print(f"📋 Available columns: {list(input_df.columns)}")
            
            # Limit processing if specified
            if max_items:
                input_df = input_df.head(max_items)
                print(f"🔄 Processing first {max_items} items")
            
            # Process each product
            enhanced_products = []
            processed_count = 0
            
            for idx, row in input_df.iterrows():
                pid = row['pid']
                
                # Skip if already processed
                if self.is_already_processed(pid):
                    print(f"⏭️  Skipping already processed PID: {pid}")
                    continue
                
                print(f"\n🔄 Processing item {idx + 1}/{len(input_df)}: {row.get('product_name', 'Unknown')[:50]}...")
                
                # Generate enhanced description
                enhanced_name, enhanced_description, product_url = self.generate_enhanced_description(row)
                
                # Create simplified product record with only 3 columns
                enhanced_product = {
                    'pid': pid,  # Keep PID for tracking processed items
                    'product_name': enhanced_name,
                    'product_url': product_url,
                    'generated_description': enhanced_description
                }
                
                enhanced_products.append(enhanced_product)
                processed_count += 1
                
                print(f"✅ Generated: {enhanced_description[:100]}...")
                
                # Save periodically (every 5 items)
                if len(enhanced_products) % 5 == 0:
                    self.save_progress(enhanced_products)
                    enhanced_products = []  # Clear the batch
                
                # Small delay to prevent overheating
                time.sleep(0.5)
            
            # Save any remaining products
            if enhanced_products:
                self.save_progress(enhanced_products)
            
            print(f"\n🎉 Processing complete! Enhanced {processed_count} products")
            print(f"💾 Results saved to: {self.output_csv_path}")
            
            # Show sample results
            if os.path.exists(self.output_csv_path):
                sample_df = pd.read_csv(self.output_csv_path).tail(3)
                print("\n📋 Sample Results:")
                for _, row in sample_df.iterrows():
                    print(f"Product: {row['product_name']}")
                    print(f"URL: {row.get('product_url', 'N/A')}")
                    print(f"Description: {row['generated_description']}")
                    print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error processing products: {e}")
            raise
    
    def save_progress(self, new_products):
        """Save progress to CSV file"""
        if not new_products:
            return
        
        try:
            # Convert to DataFrame
            new_df = pd.DataFrame(new_products)
            
            # Append to existing data or create new file
            if os.path.exists(self.output_csv_path):
                existing_df = pd.read_csv(self.output_csv_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Save to CSV with only the 3 desired columns for final output
            output_df = combined_df[['product_name', 'product_url', 'generated_description']].copy()
            output_df.to_csv(self.output_csv_path.replace('.csv', '_final.csv'), index=False)
            
            # Save full version with PID for tracking
            combined_df.to_csv(self.output_csv_path, index=False)
            
            print(f"💾 Saved progress: {len(new_df)} new items added (Total: {len(combined_df)})")
            print(f"📄 Final output (3 columns): {self.output_csv_path.replace('.csv', '_final.csv')}")
            
            # Update processed_df for duplicate checking
            self.processed_df = combined_df
            
        except Exception as e:
            print(f"❌ Error saving progress: {e}")

def main():
    # Configuration
    INPUT_CSV = "/kaggle/input/flipkart-products/flipkart_com-ecommerce_sample.csv"  # Your input CSV file name
    OUTPUT_CSV = "enhanced_flipkart_products.csv"
    MAX_ITEMS = 500  # Set to None to process all items, or specify number for testing
    
    print("🚀 Starting Flipkart Description Enhancement")
    print(f"📥 Input file: {INPUT_CSV}")
    print(f"📤 Output file: {OUTPUT_CSV}")
    print(f"📄 Final output (3 columns): {OUTPUT_CSV.replace('.csv', '_final.csv')}")
    
    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Input file '{INPUT_CSV}' not found!")
        print("Please make sure your Flipkart CSV file is in the same directory")
        return
    
    # Initialize processor
    processor = FlipkartDescriptionGenerator(INPUT_CSV, OUTPUT_CSV)
    
    # Process products
    processor.process_products(max_items=MAX_ITEMS)

if __name__ == "__main__":
    main()