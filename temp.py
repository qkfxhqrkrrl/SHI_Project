

class PGAgent_ti(nn.Module):
    def __init__(self, args: Args) -> None:
        super(PGAgent_ti, self).__init__()
        self.args = args
        self.beta = args.beta
        self.n_yard = args.n_yard
        self.n_max_block = args.n_max_block
        self.n_take_in = args.n_take_in
        self.hid_dim = args.hid_dim // 4
        self.decoder = nn.LSTMCell(input_size=5*(args.pad_len+1), hidden_size=self.hid_dim)
        self.fc_block = nn.Linear(self.hid_dim, self.n_max_block) # 400 = 40(=n_block) * 10(=n_yard)
        
        self.device = args.device
        self.batch_limit = args.barge_batch_size
        
        self.soft_max = nn.Softmax(dim=-1)
        
    
    def calculate_batch_nums(self, block_size : torch.tensor, yard_idx):
        """
        1안) 넓이로 무식하게
        2안) 왼쪽에 쭉 정렬, 오른쪽에 쭉 정렬하는 방식으로 채우자
        3안) 모든 경우의 수 다 해보고 가능한 케이스
        """
        batch_size = self.args.batch_size
        mask_l = torch.zeros_like(self.count, dtype=torch.bool) # 선택이 된 값들
        mask_l[range(batch_size), yard_idx] = True
        mask_r = torch.zeros_like(self.count, dtype=torch.bool) # count가 0인 경우
        mask_r[self.count == 0] = True
        mask = torch.logical_and(mask_l, mask_r) # 선택된 값 중 count가 0인 경우는 1로 변경
        self.count[mask] = 1 
        self.remaining_space[range(batch_size), yard_idx] -= (block_size[:, 0] * block_size[:, 1]).reshape(-1)
        
        mask_l = torch.zeros_like(self.count, dtype=torch.bool) # 선택이 된 값들
        mask_l[range(batch_size), yard_idx] = True
        mask_r = torch.zeros_like(self.count, dtype=torch.bool) # 남는 공간이 -1인 경우
        mask_r[self.remaining_space < 0] = True
        mask = torch.logical_and(mask_l, mask_r) # 선택이 된 값들 중 블록의 허용 범위가 넘어버린 경우는 카운트를 늘리고 크기는 초기화
        block_mask = torch.any(mask, axis=-1) # 값을 새로 생성할 때는 이미 마스킹이 된 상태여야 하므로 따로 개수를 확인해줘야 함
        self.count[mask] += 1
        reset_space = torch.full_like(self.remaining_space, fill_value=self.batch_limit[0] * self.batch_limit[1], dtype=torch.float32)
        reset_space[mask] -= (block_size[block_mask, 0] * block_size[block_mask, 1]).reshape(-1)
        
        return torch.sum(self.count, dim=-1)
    
    def forward(self, infos: List[pd.DataFrame]):
        """
        Feature information
        take_out, take_in, feat_vec: ["length", "width", "height", "weight", "location"]
        slot_info: [length  width  height  location  count]
        """
        batch_size = self.args.batch_size
        pad_len = self.args.pad_len
        take_in, _, yard_slots = infos
        yard_slots = torch.FloatTensor(np.expand_dims(yard_slots, axis=0).repeat(batch_size, axis=0)).to(self.device)
        
        # 돌기 전에 초기화
        self.count = torch.zeros((batch_size, self.n_yard)).to(self.device)
        self.remaining_space = torch.full((batch_size, self.n_yard), self.batch_limit[0] * self.batch_limit[1], dtype=torch.float32).to(self.device)
        
        block_mask = torch.zeros((batch_size, self.n_take_in), dtype=bool, requires_grad=False) # 제외 시키고 싶은 경우를 True
        
        probs_history = []
        rewards_history = []
        action_history = []
        
        input0 = torch.zeros(batch_size, 5*(pad_len+1)).to(self.device)
        h0, c0 = torch.zeros((batch_size, self.hid_dim), dtype=torch.float32).to(self.device), torch.zeros((batch_size, self.hid_dim), dtype=torch.float32).to(self.device)
        block_size_info = torch.FloatTensor(np.expand_dims(take_in[["length", "width", "height", "weight", "location"]].values, axis=0).repeat(batch_size, axis=0)).to(self.device)
        
        with torch.autograd.detect_anomaly():
            for idx in range(self.n_take_in):
                h0, c0 = self.decoder(input0, (h0, c0))
                out = self.fc_block(h0)
                
                # Masking
                out[:, self.n_take_in:] = -20000 # 블록 개수보다 많을 필요는 없으니 마스킹
                out[:, :self.n_take_in][block_mask] = -20000 # 마스킹 크기를 다르게
                
                
                probs = self.soft_max(out)
                m = Categorical(probs)
                action = m.sample()
                
                # 다음 스텝의 마스킹을 위해서 값들 불러오기
                block_size = block_size_info[range(batch_size), action]
                
                yard = block_size[:, -1] -1
                # print(torch.unique(yard))
                # quit()
                yard = yard.type(torch.LongTensor)
                
                slot_info = torch.stack([yard_slots[batch_idx, yard_idx] for batch_idx, yard_idx in zip(range(batch_size), yard)], dim=0)
                
                input0 = torch.cat([block_size.unsqueeze(1), slot_info], dim=1).reshape(batch_size, -1).to(self.device)
                
                # slot 중에 제일 크기가 비슷한 경우(yard_offset)을 고르고 다시 선택되지 않도록 남은 수 -1 적용
                batch_num = self.calculate_batch_nums(block_size, yard)
                
                # rewards_history.append(batch_num)
                rewards_history.append(- self.beta * batch_num)
                probs_history.append(m.log_prob(action))
                action_history.append(action)
                
                new_block_mask = block_mask.clone()
                new_block_mask[range(batch_size), action] = True
                block_mask = new_block_mask
        
        probs_history = torch.stack(probs_history).transpose(1, 0).to(self.device)
        rewards_history = torch.stack(rewards_history).transpose(1, 0).to(self.device)
        action_history = torch.stack(action_history).transpose(1, 0).to(self.device)
        
        return probs_history, rewards_history, action_history