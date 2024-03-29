import functions_framework
import torch
import cv2
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torchvision import transforms
import numpy as np 
import base64
from pymongo import MongoClient
from datetime import datetime, timezone
from to_firebase import to_firebase

class UNetModel(pl.LightningModule):
	def __init__(self, arch, encoder_name, in_channels, out_classes, cat_names, **kwargs):
		super().__init__()
		self.save_hyperparameters()
		self.model = smp.create_model(
				arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
		)

		# for image segmentation dice loss could be the best first choice
		self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
		self.epoch = 0
		self.cat_names = cat_names
    
	def calculate_stats(self, pred_masks, true_masks, iou_threshold=0.5):
		num_class_true = true_masks.size(1)
		num_det = true_masks.size(0)
		tp_class = [0] * num_class_true
		tn_class = [0] * num_class_true
		fp_class = [0] * num_class_true
		fn_class = [0] * num_class_true

		for class_true in range(num_class_true):
			pred_mask = pred_masks[:, class_true, :, :]
			true_mask = true_masks[:, class_true, :, :]
			
			pred_mask_bool = pred_mask > 0.5  
			true_mask_bool = true_mask > 0.5  # 
			
			for i in range(num_det):
				tp_class[class_true] += (pred_mask_bool[i] & true_mask_bool[i]).sum().item()
				fp_class[class_true] += (pred_mask_bool[i] & ~true_mask_bool[i]).sum().item()
				fn_class[class_true] += (~pred_mask_bool[i] & true_mask_bool[i]).sum().item()
				tn_class[class_true] += (~pred_mask_bool[i] & ~true_mask_bool[i]).sum().item()
		
		tp_tensor = torch.FloatTensor([tp_class])
		fp_tensor = torch.FloatTensor([fp_class])
		fn_tensor = torch.FloatTensor([fn_class])
		tn_tensor = torch.FloatTensor([tn_class])

		return tp_tensor, fp_tensor, fn_tensor, tn_tensor

    
	def pod(self,tp, fp, fn):    
		num_class = tp.size(1)
		num_det = tp.size(0)
		pod_total = []
	
		for e in range(num_det):
			pod_class = [0]*num_class
			for i in range(num_class):
				pod_class[i] = tp[e,i]/(2*tp[e,i]+fp[e,i] + 1e-8)
			
			pod_total.append(pod_class)

		return torch.FloatTensor(pod_total)
        
	def forward(self, image):
		mask = self.model(image)
		return mask

	def shared_step(self, batch, stage):
        
		image = batch[0]

		assert image.ndim == 4

		h, w = image.shape[2:]
		assert h % 32 == 0 and w % 32 == 0

		mask = batch[1]

		assert mask.ndim == 4

		assert mask.max() <= 1.0 and mask.min() >= 0

		logits_mask = self.forward(image)
		
		loss = self.loss_fn(logits_mask, mask)

		prob_mask = logits_mask.sigmoid()
		pred_mask = (prob_mask > 0.5).float()

		tp, fp, fn, tn = self.calculate_stats(pred_mask.long(), mask.long())

		return {
			"loss": loss,
			"tp": tp,
			"fp": fp,
			"fn": fn,
			"tn": tn,
		}

	def shared_epoch_end(self, outputs, stage):
	        
		losses = [x["loss"].detach().item() for x in outputs]
		
		tp = torch.cat([x["tp"] for x in outputs])
		fp = torch.cat([x["fp"] for x in outputs])
		fn = torch.cat([x["fn"] for x in outputs])
		tn = torch.cat([x["tn"] for x in outputs])

		per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
		
		pod_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction = "none")
		pod_score_classwise = pod_score.mean(axis = 0)
		
		dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

		try_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction = "none")
		iou_classwise = try_iou.mean(axis = 0)

		metrics = {}
		loss = sum(losses)/len(losses)
		metrics[f'{stage}_loss'] = loss
		metrics[f'{stage}_iou'] = dataset_iou
	
		for i, name in enumerate(self.cat_names):
			metrics[f'{stage}_{name}_pod'] = pod_score_classwise[i]
			metrics[f'{stage}_{name}_iou'] = iou_classwise[i]

		self.log_dict(metrics, prog_bar=True)
		return loss

	def training_step(self, batch, batch_idx):
		if batch_idx == 0:
			self.train_outputs = []
		train_cur = self.shared_step(batch, "train")
		self.train_outputs.append(train_cur)
		return train_cur           

    	def on_train_epoch_end(self):
			self.shared_epoch_end(self.train_outputs, "train")
			self.epoch += 1
			return

	def validation_step(self, batch, batch_idx):
		if batch_idx == 0:
			self.valid_outputs = []
		valid_cur = self.shared_step(batch, "valid")
		self.valid_outputs.append(valid_cur)
		return valid_cur   

	def on_validation_epoch_end(self):
		return self.shared_epoch_end(self.valid_outputs, "valid")

	def test_step(self, batch, batch_idx):
		if batch_idx == 0:
			self.test_outputs = []
		test_cur = self.shared_step(batch, "test")
		self.test_outputs.append(test_cur)
		return test_cur  

	def on_test_epoch_end(self):
		return self.shared_epoch_end(self.test_outputs, "test")

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=0.0001)



def apply_low_pass_filter(image, sigma=50):
	# Convierte la imagen a espacios de color HSV
	
	# Obtiene los canales de color por separado
	hue, saturation, value = cv2.split(image)
	
	# Aplica el filtro a cada canal por separado
	for channel in [hue, saturation, value]:
		# Calcula la transformada de Fourier del canal
		f_transform = np.fft.fft2(channel)
		fshift = np.fft.fftshift(f_transform)
	
		# Crea un filtro pasa bajos suavizado (filtro gaussiano)
		rows, cols = channel.shape
		crow, ccol = rows // 2, cols // 2
		x = np.linspace(-ccol, ccol, cols)
		y = np.linspace(-crow, crow, rows)
		X, Y = np.meshgrid(x, y)
		mask = np.exp(-((X ** 2 + Y ** 2) / (2 * sigma ** 2)))
	
		# Aplica el filtro
		fshift = fshift * mask
	
		# Calcula la inversa de la transformada de Fourier
		f_ishift = np.fft.ifftshift(fshift)
		img_back = np.fft.ifft2(f_ishift)
		img_back = np.abs(img_back)
	
		# Escala los valores al rango 0-255
		img_back_scaled = np.uint8(img_back * 255 / np.max(img_back))
	
		# Actualiza el canal con la imagen filtrada
		channel[:] = img_back_scaled

    # Fusiona los canales nuevamente
	filtered_image_hsv = cv2.merge((hue, saturation, value))

    # Convierte la imagen de vuelta a BGR
    #	filtered_image_bgr = cv2.cvtColor(filtered_image_hsv, cv2.COLOR_HSV2BGR)

	return filtered_image_hsv 


def in_polygon(cnt, w_in, h_in, mask):
	list_value = list()
	for x in range(w_in):
		for y in range(h_in):
			result = cv2.pointPolygonTest(cnt, (x,y), False) 
			if result==1:
				list_value.append(mask[y,x])
	return list_value


def avr_images(l, idx, color_pos, porc, label_colors, color_canal):
    true_indices = np.where(idx)  # Obtener los índices de True en idx
    for i in range(len(true_indices[0])):
        row = true_indices[0][i]
        col = true_indices[1][i]
        label_color = label_colors[l, color_pos]
        color = color_canal[row, col]
        updated_color = label_color * (1 - porc) + color * porc
        color_canal[row, col] = updated_color

    return color_canal

def decode_segmap(tensor, img_in, choosen_cat_names, choose_class, threshold = 0.5, porc=0.5):
  
	label_colors = np.array([(123, 80,  50)])
	
	h_in, w_in, _ = img_in.shape
	
	r = np.zeros((h_in, w_in)).astype(np.uint8)
	g = np.zeros((h_in, w_in)).astype(np.uint8)
	b = np.zeros((h_in, w_in)).astype(np.uint8)
	
	r_in = img_in[:,:,0]
	g_in = img_in[:,:,1]
	b_in = img_in[:,:,2]
	
	results = list()

	for l in reversed(range(tensor.size(1))):
		if choosen_cat_names[l] in choose_class:
			mask = (tensor[0,l,:,:].numpy()*255).astype(np.uint8)
			mask = cv2.resize(mask, (w_in, h_in))
			filtred_mask = np.zeros(mask.shape, dtype=np.uint8)
			ret, bin_mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
			
			cnts = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnts = cnts[0] if len(cnts) == 2 else cnts[1]
			
			red = label_colors[l, 0].item()
			green = label_colors[l, 1].item()
			blue = label_colors[l, 2].item()
			
			color = (red,green,blue)
			
			h, w = bin_mask.shape
			
			normalized_cnts = []
			
			for cnt in cnts:
				lado = mask.shape[0]
				cant_pixel = 0
			
				extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
				
				list_value = in_polygon(cnt, w_in, h_in, mask)
				
				promedio = sum(list_value)/(len(list_value)*255 + 1e-8)
				
				if promedio > threshold[choosen_cat_names[l]]:
					cv2.fillPoly(filtred_mask, pts=[cnt], color= 255)
				
					result = {"mean":promedio, 
						  "clase":choosen_cat_names[l], 
						  "coord":extTop, 
						  "color":color,
						  "cnt":cnt}
					
					results.append(result)

			filtred_true_mask = filtred_mask>0
			idx = filtred_true_mask == True
			
			r[idx] = int(red)
			g[idx] = int(green)
			b[idx] = int(blue)
			
			r_in = avr_images(l, idx, 0, porc, label_colors, r_in)
			g_in = avr_images(l, idx, 1, porc, label_colors, g_in)
			b_in = avr_images(l, idx, 2, porc, label_colors, b_in)
			
			rgb = np.stack([r, g, b], axis=2)
			rgb_in_out = np.stack([r_in, g_in, b_in], axis=2)
			
			for result_ in results:
			
				x_min = result_['coord'][0] - 25
				if x_min<0:
					x_min=0
				    
				y_min = result_['coord'][1] - 5
				
				mean = round(result_['mean'], 3)
				
				rgb_in_out = cv2.polylines(rgb_in_out, [result_['cnt']], True, result_['color'], 2)
				
				((text_w, text_h), _) = cv2.getTextSize(result_['clase'] + " " + str(mean), 
									cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
				
				rgb_in_out = cv2.rectangle(rgb_in_out, 
							    (x_min, y_min - int(1.3 * text_h)), 
							    (x_min + text_w, y_min), 
							    result_['color'], -1)
				
				rgb_in_out = cv2.putText(rgb_in_out, 
							result_['clase'] + " " + str(mean),
							(x_min, y_min - int(0.3 * text_h)), 
							cv2.FONT_HERSHEY_SIMPLEX, 
							1, (255, 255, 255), 2)
                
    return rgb, rgb_in_out, results


model = UNetModel("unet", "efficientnet-b0", in_channels=3, out_classes=1, cat_names = ['water'])
checkpoint = torch.load("best.ckpt", map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['state_dict'])


connection_string = "mongodb://deliryum-mongodb:Z50r80Z1WKtYhEd2hvt5vcGwDFCgF50GiYaPOOyIEpO8y2cXeODMqRTiz47hek5PG0ja4KjC4hwVACDbFJv2GQ==@deliryum-mongodb.mongo.cosmos.azure.com:10255/?ssl=true&retrywrites=false&replicaSet=globaldb&maxIdleTimeMS=120000&appName=@deliryum-mongodb@"

client = MongoClient(connection_string)
conn = client.get_database("deliryum")

@functions_framework.http
def main(request):
	now = datetime.now()
	fecha_actual =  datetime(now.year, now.month, now.day, now.hour, now.minute, now.second, tzinfo=timezone.utc)
	
	collection_conn = conn.get_collection("prueba")
	
	request_json = request.get_json(silent=True)
	request_args = request.args
	
	image_data = request_json.get('img')
	lugar = request_json.get('lugar')
	pixel_metro = request_json.get('pixel_metro')
	
	remote_path_img = f"{lugar}_{str(fecha_actual)}.jpg"
	firebase = to_firebase(remote_path_img)
	
	image_bytes = base64.b64decode(image_data)
	
	img_ori = np.frombuffer(image_bytes, dtype=np.uint8)
	try:
		img_ori = cv2.imdecode(img_ori, cv2.IMREAD_COLOR)
		firebase.start(img_ori)
		print(img_ori)
		#img_ori = normalize(img_ori)
		img_proc = apply_low_pass_filter(img_ori, 40)
		
		print(img_ori.shape)
		img = cv2.resize(img_proc, (512, 512))
		
		transform = transforms.ToTensor()
		tensor = transform(img)
		
		tensor = tensor.unsqueeze(0)
		model.eval()  

		with torch.no_grad():
    		logits = model(tensor)
        
        	pr_masks = logits.sigmoid()

        	pr_masks, img_pls_mask, results = decode_segmap(pr_masks, img_ori.copy(), ['water'], ['water'],threshold = {'water':0.9}, porc = 0.5)

        	area = 0

        	for result in results:
            	area += cv2.contourArea(result['cnt'])

        	output = {"area": area, 
			  "path_img": remote_path_img, 
			  "fecha_iso": fecha_actual.isoformat(), 
			  "lugar": lugar,
			  "pixel_metro": pixel_metro,
			  "fecha_filtro": str(fecha_actual)}

        	response = collection_conn.insert_one(output)

		return "OK"
	except:
		return "ERROR" 
